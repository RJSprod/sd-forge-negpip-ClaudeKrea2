"""NegPiP support for Forge Neo's Krea 2 diffusion model.

Krea 2 encodes the prompt with Qwen3-VL, then concatenates the resulting text
tokens with the reference and image streams inside a single transformer.
NegPiP reaches that transformer by one of two methods.

"Value flip" is what NegPiP does on SD, and what every other Krea 2 NegPiP
does: the negated words are encoded at their magnitude, so the conditioning
stays a faithful representation of them, and their sign is carried as a mask
that negates their attention values in every single-stream block.  Attention
still routes to the concept, and the image subtracts it instead of adding it.
That removes a concept the prompt would have drawn.  It cannot remove one a
reference image puts in front of the model: in Edit mode the reference is
hundreds of vision tokens inside the prompt and thousands of latent tokens
beside it, against which two text tokens with their sign flipped are nothing.

"Attention guidance", the default, is Normalized Attention Guidance (Chen et
al., 2025, https://arxiv.org/abs/2505.21179), which restores negative
prompting on few-step models at CFG 1 and is proven on reference editing.
The prompt is encoded twice: without the negated words, as the positive
context, and the negated words alone, beside the same reference images, as
the negative one.  Both ride in the text stream, kept apart in the text
fusion transformer, and in every single-stream block the reference and image
tokens read both attentions and are steered away from the negative one, the
difference normalized so it cannot blow up.  The two contexts share the image
tokens, as NAG's own single-stream implementation does, so nothing is batched
twice and Forge's handling of the reference latents is untouched.  The weight
sets the guidance scale.

The text engine, a port of ComfyUI's since Forge Neo rewrote them all, does
not weight the prompt at all.  It tokenizes the templated prompt with weights
disabled, so `(foo:-1.0)` reaches the language model as literal characters,
which spell out the very concept that was meant to be removed.  The hooks
below parse the weights themselves and tokenize every region on its own,
between a single copy of the chat template, so a weight lands on the words
alone.  The emphasis is applied to the input embeddings once the vision tower
has expanded every reference image, as Forge's own engine used to; the
engine is left to encode, strip the template and reshape exactly as it would.

Both methods reach the transformer through the override Forge's attention
function accepts, which every attention backend honours, so nothing inside
the blocks is reimplemented.
"""

from functools import partial, wraps
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

import torch

from backend import attention
from backend.args import dynamic_args
from backend.sampling import condition, sampling_function
from modules import shared

from .utils import KREA2_FLIP, KREA2_GUIDANCE, krea2_settings

ENGINE_METHODS: tuple[str, ...] = ("__call__", "_tokenize_with_weights")
ENGINE_REQUIREMENTS: tuple[str, ...] = (
    "_tokenize_with_weights",
    "text_encoder.process_tokens",
    "tokenizer.tokenizer",
)

IMAGE_PAD: int = 151655
"""`<|image_pad|>`, which the engine swaps for the image it stands for"""

NEGATIVE_KEY: str = "\x00negpip:negative"
"""appended to a prompt to name the encoding of its negated words"""

SCALE_PER_WEIGHT: float = 4.0
"""guidance scale per unit of weight, so that `-1.0` is NAG's default of 5"""


class _EngineState:
    """Carries the per-prompt data between the patched text engine methods."""

    engine_cls: type = None
    """the patched text engine class"""

    encoder = None
    """the text encoder whose `process_tokens` is hooked"""

    emphasis = None
    """emphasis of the engine call in progress, as configured by the user"""

    plans: dict[str, list[list]] = None
    """the regions to tokenize, by the prompt the engine is handed"""

    pending: tuple[str, list[float]] = None
    """the prompt being encoded, and the weights of its tokens"""

    aligned: dict[str, list[float]] = None
    """^ the same weights per prompt, expanded over the embeddings of every image"""

    masks: list[torch.Tensor] = None
    """sign masks of the last engine call, one per prompt"""

    counts: list[int] = None
    """^ how many tokens each of them negates"""

    method: str = KREA2_GUIDANCE
    """the Krea 2 method, as configured when the model was patched"""

    tau: float = 2.5
    alpha: float = 0.25
    """the guidance parameters, likewise"""

    dit_cls: type = None
    """the patched transformer class"""

    reported: bool = False
    """whether this generation has confirmed NegPiP reaching attention"""

    def reset(self):
        self.emphasis = None
        self.plans = None
        self.pending = None
        self.aligned = None
        self.masks = None
        self.counts = None
        self.reported = False


_state = _EngineState()
_state.reset()
_originals: dict[str, Callable] = {}
_dit_originals: dict[str, Callable] = {}


def patch_krea2_negpip(cls: "NegPiP", model=None, *, unpatch: bool = False):
    """Install or remove all Krea 2 hooks.

    Hooks are stored on the objects they modify so model changes and failed or
    interrupted generations can safely call this function more than once.

    The caller passes the model it decided was Krea 2, so that the hooks cannot
    land on a different one than the check was made against.
    """
    if unpatch != cls._patched[2]:
        return

    if unpatch:
        model = getattr(cls, "_krea2_patched_model", None)
    elif model is None:
        model = shared.sd_model
    if model is None:
        cls._patched[2] = False
        return
    dit = model.forge_objects.unet.model.diffusion_model
    if unpatch:
        _unhook(model, dit)
        del cls._krea2_patched_model
        cls._patched[2] = False
        return

    # Validate the model before changing any global methods: a transformer
    # whose attention cannot be reached should fail here rather than mid
    # sampling.
    engine = _find_text_engine(model)
    _probe_override(dit)
    _state.method, _state.tau, _state.alpha = krea2_settings()
    cls._krea2_patched_model = model
    try:
        _hook_engine(engine, False)
        _hook_get_learned_conditioning(model, False)
        _hook_dit_forward(dit, False)
        _hook_compile_conditions(False)
    except Exception:
        # Each remover tolerates a partially installed hook set.
        _unhook(model, dit)
        del cls._krea2_patched_model
        raise
    cls._patched[2] = True


def _unhook(model, dit):
    _hook_get_learned_conditioning(model, True)
    _hook_dit_forward(dit, True)
    _hook_compile_conditions(True)
    _hook_engine(None, True)
    _state.reset()


def _find_text_engine(model):
    for name in (
        "text_processing_engine_qwen",
        "text_processing_engine_qwen3_vl",
        "text_processing_engine_qwen3vl",
        "text_processing_engine_krea2",
        "text_processing_engine",
    ):
        engine = getattr(model, name, None)
        if engine is not None:
            return engine

    for name, engine in vars(model).items():
        if name.startswith("text_processing_engine") and engine is not None:
            return engine

    raise RuntimeError("NegPiP could not find Krea 2's Qwen3-VL text engine")


# ================================================================================ #
# Text Engine


def _hook_engine(engine, remove: bool):
    """Patch the text engine class, as `__call__` bypasses the instance.

    The encoder it wraps is hooked on the instance instead, since its class
    also serves the text engine of every other model.
    """

    if remove:
        for name, original in _originals.items():
            if getattr(_state.engine_cls, name, None) is not None:
                setattr(_state.engine_cls, name, original)
        _originals.clear()
        if _state.encoder is not None:
            _hook_process_tokens(_state.encoder, True)
        _state.engine_cls = None
        _state.encoder = None
        return

    engine_cls = type(engine)

    for path in ENGINE_REQUIREMENTS:
        found = engine
        for attr in path.split("."):
            found = getattr(found, attr, None)
        if not callable(found):
            raise RuntimeError(f"NegPiP could not find Krea 2's {path}")
    # the prompt is templated by hand below, so the slot has to be there
    if "{}" not in getattr(engine, "llama_template", ""):
        raise RuntimeError("NegPiP could not find Krea 2's prompt template")

    _state.engine_cls = engine_cls

    replacements = {
        "__call__": _negpip_call,
        "_tokenize_with_weights": _negpip_tokenize_with_weights,
    }

    for name in ENGINE_METHODS:
        _originals[name] = getattr(engine_cls, name)
        setattr(engine_cls, name, replacements[name])

    _state.encoder = engine.text_encoder
    _hook_process_tokens(engine.text_encoder, False)


def _negpip_call(self, texts: list[str], images: list[torch.Tensor] = []):
    """Encode every prompt as the engine does, keeping its emphasis weights.

    Under attention guidance a prompt with negated words is encoded twice,
    without them and as them alone, and the two are concatenated; the mask
    tells the transformer where one ends and the other begins.
    """

    from backend.text_processing import emphasis

    option = emphasis.get_current_option(shared.opts.emphasis)()
    emphasized = any(emphasis.uses_emphasis(x) for x in texts)

    # "None" reads a weight as literal characters, so there is no region to
    # negate, and the engine is left to encode the prompt exactly as it would.
    # Every other mode parses them, and the sign is all NegPiP needs: the
    # magnitude only decides how strongly the concept is emphasized first.
    if isinstance(option, emphasis.EmphasisNone):
        if emphasized:
            print(f'NegPiP Disabled (Emphasis: "{option.name}")')
        zs = _originals["__call__"](self, texts, images)
        _state.masks = [_sign_mask(None, z) for z in zs]
        _state.counts = [0] * len(zs)
        return zs

    guidance = _state.method == KREA2_GUIDANCE
    plans: dict[str, list[list]] = {}
    lines: list[tuple[str, Optional[str], float, int]] = []

    for line in texts:
        regions = _regions(line, option.name)
        negatives = [text.strip() for text, weight in regions if weight < 0]

        if guidance and negatives:
            key = line + NEGATIVE_KEY
            phrase = ", ".join(text for text in negatives if text)
            plans[line] = _without_negatives(regions)
            plans[key] = [[phrase, 1.0]]
            weight = max(-weight for _, weight in regions if weight < 0)
            count = len(self.tokenizer.tokenizer(phrase)["input_ids"])
            lines.append((line, key, weight, count))
        else:
            plans[line] = regions
            lines.append((line, None, 0.0, 0))

    _state.emphasis = option
    _state.plans = plans
    _state.aligned = {}
    try:
        zs = _originals["__call__"](self, list(plans), images)
        aligned = _state.aligned
    finally:
        _state.emphasis = None
        _state.plans = None
        _state.pending = None
        _state.aligned = None

    if not isinstance(zs, list) or len(zs) != len(plans):
        raise RuntimeError("Unexpected Krea 2 encodings")
    encoded = dict(zip(plans, zs))

    # the engine records "None", which is how it would have read them itself
    if emphasized:
        dynamic_args.last_extra_generation_params["Emphasis"] = option.name

    conds: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    counts: list[int] = []

    for line, key, weight, count in lines:
        for encoding in (line, key):
            # a silent no-op is the very failure this module exists to avoid
            if encoding is not None and encoding not in aligned:
                raise RuntimeError("NegPiP could not align Krea 2's emphasis weights")

        z = encoded[line]
        if key is None:
            mask = _sign_mask(aligned[line], z)
            count = int((mask < 0).sum())
        else:
            negative = encoded[key]
            mask = torch.cat(
                (z.new_ones(len(z)), negative.new_full((len(negative),), -weight))
            )
            mask = mask.reshape(-1, 1, 1)
            z = torch.cat((z, negative))

        conds.append(z)
        masks.append(mask)
        counts.append(count)

    _state.masks = masks
    _state.counts = counts
    return conds


def _regions(line: str, emphasis: str) -> list[list]:
    """The prompt as the regions of one templated conversation.

    Upstream tokenizes the templated prompt as a whole, with weights disabled.
    Each region is tokenized on its own here instead, between a single copy of
    the template, which keeps the prompt well formed and puts the weight on
    the words alone.
    """

    from backend.text_processing import parsing

    regions: list[list] = [
        [text, weight]
        for text, weight in parsing.parse_prompt_attention(line, emphasis)
        # a chunking hint for CLIP, which parses to a weight of -1 and would
        # otherwise be negated as if it were part of the prompt
        if text != "BREAK"
    ]
    if not regions:
        regions = [["", 1.0]]

    # upstream strips the prompt before templating it
    regions[0][0] = regions[0][0].lstrip()
    regions[-1][0] = regions[-1][0].rstrip()

    return regions


def _without_negatives(regions: list[list]) -> list[list]:
    """The regions of the positive context: the negated ones taken out."""

    kept: list[list] = []
    removed = False

    for text, weight in regions:
        if weight < 0:
            # the whitespace that set the negated words apart goes with them
            if kept:
                kept[-1][0] = kept[-1][0].rstrip()
            removed = True
            continue
        if removed:
            text = text.lstrip()
            removed = False
        kept.append([text, weight])

    if not kept:
        kept = [["", 1.0]]

    return kept


def _negpip_tokenize_with_weights(self, text: str, images: list[torch.Tensor] = []):
    """Tokenize the regions planned for a prompt, keeping their weights.

    The encoder is handed a weight of 1.0 throughout, since it would apply
    anything else across the layers rather than the tokens, and the weights
    themselves are kept for `process_tokens` to apply.
    """

    plans = _state.plans
    if plans is None or text not in plans:
        # not called by NegPiP's engine call, so there is nothing to keep
        return _originals["_tokenize_with_weights"](self, text, images)

    regions = plans[text]

    prefix, _, suffix = self.llama_template.partition("{}")
    if images:
        prefix += self.vision_block * len(images)

    texts = [prefix, *(region for region, _ in regions), suffix]
    weights = [1.0, *(weight for _, weight in regions), 1.0]

    pairs: list[tuple] = []
    multipliers: list[float] = []
    embed_count = 0

    for tokens, weight in zip(self.tokenizer.tokenizer(texts)["input_ids"], weights):
        for token in tokens:
            if token == IMAGE_PAD and embed_count < len(images):
                token = {
                    "type": "image",
                    "data": images[embed_count],
                    "original_type": "image",
                }
                embed_count += 1

            pairs.append((token, 1.0))
            multipliers.append(weight)

    _state.pending = (text, multipliers)
    return [pairs]


def _hook_process_tokens(encoder, remove: bool):
    """Weight the input embeddings, once every image has been expanded."""

    if remove:
        original = getattr(encoder, "negpip_orig_process_tokens", None)
        if original is not None:
            if getattr(encoder.process_tokens, "_negpip_krea2", False):
                encoder.process_tokens = original
            del encoder.negpip_orig_process_tokens
        return

    original = encoder.process_tokens
    encoder.negpip_orig_process_tokens = original

    @wraps(original)
    def process_tokens(tokens, device):
        result = original(tokens, device)

        pending, _state.pending = _state.pending, None
        if pending is None or len(tokens) != 1:
            return result
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError("Unexpected Krea 2 embeddings")

        line, multipliers = pending
        embeds, attention_mask, num_tokens, embeds_info = result
        aligned = _align_multipliers(
            tokens[0], multipliers, embeds_info, embeds.shape[1]
        )
        if aligned is None:
            return result
        _state.aligned[line] = aligned

        # the emphasis is handed the magnitudes: scaling an input embedding by
        # a negative weight encodes as no word at all, so the sign is left to
        # the mask and applied in attention instead
        magnitudes = torch.tensor(
            [[abs(weight) for weight in aligned]],
            device=embeds.device,
            dtype=embeds.dtype,
        )
        weighted = _state.emphasis(embeds, magnitudes)
        if weighted is not None:  # "Ignore" leaves them as they are
            embeds = weighted

        return embeds, attention_mask, num_tokens, embeds_info

    process_tokens._negpip_krea2 = True
    encoder.process_tokens = process_tokens


def _align_multipliers(
    tokens: list,
    multipliers: list[float],
    embeds_info: Optional[list[dict]],
    length: int,
) -> Optional[list[float]]:
    """Expand the weights over the embeddings that every image turns into."""

    if len(tokens) != len(multipliers):
        return None

    weights = [
        w for token, w in zip(tokens, multipliers) if not isinstance(token, dict)
    ]
    images = [w for token, w in zip(tokens, multipliers) if isinstance(token, dict)]

    inserts = [
        info
        for info in (embeds_info or ())
        if isinstance(info, dict) and "index" in info and "size" in info
    ]

    if len(inserts) != len(images):
        # an embed was dropped, so which image was skipped is unknown; the
        # vision block is never emphasized anyway
        images = [1.0] * len(inserts)

    for weight, info in zip(images, inserts):
        index = int(info["index"])
        weights[index:index] = [weight] * int(info["size"])

    return weights if len(weights) == length else None


def _sign_mask(weights: Optional[list[float]], z: torch.Tensor) -> torch.Tensor:
    """Build the value mask of a prompt, aligned with its conditioning.

    Negated tokens keep their weight rather than collapsing to -1, so it is a
    dial rather than a switch.  It needs to be: a single-stream transformer
    attends over the reference and image streams as well, so a text token is a
    far smaller share of the attention than it is in the cross attention of SD,
    and `-1.0` alone is easily too little to see.  Everything else stays at 1.
    """

    length = z.shape[0]
    ones = torch.ones(length, 1, 1, device=z.device, dtype=z.dtype)

    if not weights or len(weights) < length:
        return ones

    # the template of the first region was stripped off the front
    values = torch.as_tensor(weights[-length:], device=z.device, dtype=z.dtype)
    values = values.reshape(length, 1, 1)
    return torch.where(values < 0.0, values, ones)


# ================================================================================ #
# Conditioning


def _hook_get_learned_conditioning(model, remove: bool):
    if remove:
        original = getattr(model, "negpip_orig_get_learned_conditioning", None)
        if original is not None:
            if getattr(model.get_learned_conditioning, "_negpip_krea2", False):
                model.get_learned_conditioning = original
            del model.negpip_orig_get_learned_conditioning
        return

    original = model.get_learned_conditioning
    model.negpip_orig_get_learned_conditioning = original

    @torch.inference_mode()
    @wraps(original)
    def get_learned_conditioning(prompt):
        _state.masks = None
        _state.counts = None
        _state.reported = False
        conds = original(prompt)
        masks, _state.masks = _state.masks, None
        counts, _state.counts = _state.counts, None

        # Krea 2 returns one [tokens, layers, features] tensor per prompt.
        # Keep this deliberately strict: silently losing the mask is worse than
        # exposing an upstream conditioning API change.
        if not isinstance(conds, list) or len(conds) != len(prompt):
            raise RuntimeError("Unexpected Krea 2 conditioning result")
        if masks is None or len(masks) != len(conds):
            raise RuntimeError("Unexpected Krea 2 conditioning masks")
        if counts is None or len(counts) != len(conds):
            raise RuntimeError("Unexpected Krea 2 conditioning masks")

        contexts, negpip_masks = [], []
        count = 0

        for cond, mask, negated in zip(conds, masks, counts):
            if not isinstance(cond, torch.Tensor):
                raise RuntimeError("Unexpected Krea 2 conditioning item")

            mask = mask.to(cond)
            count += int(negated)
            # the conditioning is left alone: it is what the text fusion
            # transformer reads, and the concept has to survive it intact for
            # attention to route to the tokens the mask then subtracts
            contexts.append(cond)
            negpip_masks.append(mask)

        if count:
            key = (
                "Negative"
                if getattr(prompt, "is_negative_prompt", False)
                else "Positive"
            )
            print(f"NegPiP Enable ({key}: {count})")

        # lists, so that prompts of different lengths are padded downstream
        return {"crossattn": contexts, "c_negpip_mask": negpip_masks}

    get_learned_conditioning._negpip_krea2 = True
    model.get_learned_conditioning = get_learned_conditioning


def _hook_compile_conditions(remove: bool):
    if remove:
        original = getattr(condition, "negpip_krea2_orig_compile_conditions", None)
        if original is not None:
            condition.compile_conditions = original
            sampling_function.compile_conditions = original
            del condition.negpip_krea2_orig_compile_conditions
        return

    original = condition.compile_conditions
    condition.negpip_krea2_orig_compile_conditions = original

    @wraps(original)
    def compile_conditions(cond):
        if isinstance(cond, dict) and "crossattn" in cond and "vector" not in cond:
            crossattn = cond["crossattn"]
            model_conds = {"c_crossattn": condition.ConditionCrossAttn(crossattn)}
            if "c_negpip_mask" in cond:
                model_conds["c_negpip_mask"] = condition.Condition(
                    cond["c_negpip_mask"]
                )
            return [dict(cross_attn=crossattn, model_conds=model_conds)]
        return original(cond)

    compile_conditions._negpip_krea2 = True
    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions


# ================================================================================ #
# Transformer


def _probe_override(dit):
    """Check that the transformer's attention can be intercepted.

    The blocks are not reimplemented; both methods reach them through the
    override Forge's attention function accepts in its `transformer_options`,
    whichever backend is behind it.
    """

    for name in ("txtfusion", "blocks"):
        if getattr(dit, name, None) is None:
            raise RuntimeError(f"NegPiP could not find Krea 2's {name}")

    marker = object()
    probe = torch.zeros(1, 1, 1, 8)
    try:
        seen = attention.attention_function(
            probe,
            probe,
            probe,
            1,
            skip_reshape=True,
            transformer_options={
                "optimized_attention_override": lambda *a, **k: marker
            },
        )
    except Exception as error:
        raise RuntimeError(
            f"NegPiP could not probe Forge's attention: {error}"
        ) from error
    if seen is not marker:
        raise RuntimeError("NegPiP could not find Forge's attention override")


def _hook_dit_forward(dit, remove: bool):
    """Patch the transformer class rather than the instance we were handed.

    `forge_objects` is rebuilt from `forge_objects_after_applying_lora` before
    every sampling, and the patcher restores its object patches around it, so
    an attribute set on one instance at `process_batch` time is not reliably
    the attribute that runs.  Patching the class always runs, and hands the
    attention override to whichever instance actually arrives.
    """

    if remove:
        cls = _state.dit_cls
        original = _dit_originals.pop("forward", None)
        if cls is not None and original is not None:
            if getattr(cls.forward, "_negpip_krea2", False):
                cls.forward = original
        _state.dit_cls = None
        return

    cls = type(dit)
    original = cls.forward
    _dit_originals["forward"] = original
    _state.dit_cls = cls

    @torch.inference_mode()
    @wraps(original)
    def forward(self, *args, **kwargs):
        mask = _attention_mask(kwargs.pop("c_negpip_mask", None))
        if mask is None:
            return original(self, *args, **kwargs)

        layers = int(getattr(self, "txtlayers", 0))
        args, kwargs = _with_override(args, kwargs, mask, layers)
        return original(self, *args, **kwargs)

    forward._negpip_krea2 = True
    cls.forward = forward


def _attention_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Reshape the conditioning mask to the [batch, tokens, 1] of attention."""

    if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
        return None

    batch, tokens = mask.shape[0], mask.shape[1]
    if mask.numel() != batch * tokens:
        return None
    if not bool((mask < 0).any()):
        return None

    return mask.reshape(batch, tokens, 1)


def _with_override(args: tuple, kwargs: dict, mask: torch.Tensor, layers: int):
    """Hand the transformer a copy of its options carrying the override."""

    if "transformer_options" in kwargs:
        options = dict(kwargs["transformer_options"] or {})
        kwargs = {**kwargs, "transformer_options": options}
    elif len(args) > 4:
        options = dict(args[4] or {})
        args = (*args[:4], options, *args[5:])
    else:
        options = {}
        kwargs = {**kwargs, "transformer_options": options}

    previous = options.get("optimized_attention_override", None)
    options["optimized_attention_override"] = _attention_override(
        mask, layers, previous
    )
    return args, kwargs


def _layouts(mask: torch.Tensor) -> list[tuple[int, int, float]]:
    """How each prompt's text stream splits: positive tokens, negative, scale."""

    layouts = []
    for row in mask[:, :, 0]:
        negative = row < 0
        count = int(negative.sum())
        if count == 0:
            layouts.append((row.shape[0], 0, 1.0))
            continue
        if not bool(negative[-count:].all()) or bool(negative[:-count].any()):
            raise RuntimeError("Krea 2 NegPiP mask does not match the method")
        weight = float(-row[-1])
        layouts.append((row.shape[0] - count, count, 1.0 + SCALE_PER_WEIGHT * weight))

    return layouts


def _attention_override(mask: torch.Tensor, layers: int, previous: Optional[Callable]):
    """Build the attention override of one transformer forward.

    `mask` is [batch, text tokens, 1], the sign and weight of every token of
    the text stream.  Forge's attention function hands the override its own
    implementation along with the arguments it was called with, for every
    attention inside the transformer: the text fusion, whose layerwise blocks
    attend over the encoder layers of one token at a time and whose refiners
    attend over the text, and the single-stream blocks over the whole stream.
    """

    method = _state.method
    batch, text_length = mask.shape[0], mask.shape[1]
    layouts = _layouts(mask) if method == KREA2_GUIDANCE else None

    def override(func, q, k, v, heads, *args, **kwargs):
        attend = func if previous is None else partial(previous, func)

        if (
            args
            or kwargs.get("mask", None) is not None
            or not kwargs.get("skip_reshape", False)
            or q.ndim != 4
        ):
            return attend(q, k, v, heads, *args, **kwargs)
        if q.shape[2] == layers and q.shape[0] == batch * text_length:
            return attend(q, k, v, heads, **kwargs)  # layerwise text fusion
        if q.shape[0] != batch or q.shape[2] < text_length:
            raise RuntimeError("Krea 2 NegPiP mask does not match attention")

        if method == KREA2_FLIP:
            return _flip(attend, q, k, v, heads, kwargs, mask)
        return _guide(attend, q, k, v, heads, kwargs, layouts, text_length)

    return override


def _flip(attend, q, k, v, heads, kwargs, mask):
    """Negate the values of the negated tokens, for the whole stream to read."""

    text_length = mask.shape[1]
    if q.shape[2] == text_length:
        return attend(q, k, v, heads, **kwargs)  # the text fusion refiners

    values = v.clone()
    values[:, :, :text_length] *= mask.unsqueeze(1).to(values)

    if not _state.reported:
        _state.reported = True
        negated = int((mask < 0).sum())
        print(
            f"NegPiP Applied (Krea 2 value flip: {negated} of {text_length} text "
            f"tokens, {q.shape[2]} in the stream)"
        )

    return attend(q, k, values, heads, **kwargs)


def _guide(attend, q, k, v, heads, kwargs, layouts, text_length):
    """Steer the reference and image tokens away from the negative context."""

    if len(set(layouts)) > 1:
        # prompts laid out differently were batched together; one at a time
        return torch.cat(
            [
                _guide(
                    attend,
                    q[i : i + 1],
                    k[i : i + 1],
                    v[i : i + 1],
                    heads,
                    kwargs,
                    [layout],
                    text_length,
                )
                for i, layout in enumerate(layouts)
            ]
        )

    positive, negative, scale = layouts[0]
    if negative == 0:
        return attend(q, k, v, heads, **kwargs)

    if q.shape[2] == text_length:
        # the text fusion refiners: the two contexts attend apart
        out_positive = attend(
            q[:, :, :positive], k[:, :, :positive], v[:, :, :positive], heads, **kwargs
        )
        out_negative = attend(
            q[:, :, positive:], k[:, :, positive:], v[:, :, positive:], heads, **kwargs
        )
        return torch.cat((out_positive, out_negative), dim=1)

    # the stream is [positive context, negative context, references, image];
    # the references and image read both contexts, on their own
    def beside(t):
        return torch.cat((t[:, :, :positive], t[:, :, text_length:]), dim=2)

    out_positive = attend(beside(q), beside(k), beside(v), heads, **kwargs)
    out_negative = attend(
        q[:, :, positive:], k[:, :, positive:], v[:, :, positive:], heads, **kwargs
    )
    guided = _nag(
        out_positive[:, positive:],
        out_negative[:, negative:],
        scale,
        _state.tau,
        _state.alpha,
    )

    if not _state.reported:
        _state.reported = True
        print(
            f"NegPiP Applied (Krea 2 attention guidance: scale {scale:g}, {negative} "
            f"negative tokens beside {positive}, {q.shape[2] - text_length} in the image stream)"
        )

    return torch.cat(
        (out_positive[:, :positive], out_negative[:, :negative], guided), dim=1
    )


def _nag(
    positive: torch.Tensor,
    negative: torch.Tensor,
    scale: float,
    tau: float,
    alpha: float,
) -> torch.Tensor:
    """Normalized Attention Guidance over one attention output.

    After ChenDarYen/ComfyUI-NAG (MIT): extrapolate from the negative attention
    past the positive one, cap the L1 norm of the result at `tau` times the
    positive's so it cannot blow up, and blend `alpha` of it into the positive.
    """

    guidance = positive * scale - negative * (scale - 1.0)

    norm_positive = torch.linalg.vector_norm(
        positive, ord=1, dim=-1, keepdim=True, dtype=torch.float32
    )
    norm_guidance = torch.linalg.vector_norm(
        guidance, ord=1, dim=-1, keepdim=True, dtype=torch.float32
    )
    ratio = norm_guidance.clamp_min(1e-6) / norm_positive.clamp_min(1e-6)
    guidance = guidance * (ratio.clamp(max=tau) / ratio).to(guidance)

    return guidance * alpha + positive * (1.0 - alpha)
