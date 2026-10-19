"""NegPiP support for Forge Neo's Krea 2 diffusion model.

Krea 2 encodes the prompt with Qwen3-VL, then concatenates the resulting text
tokens with the reference and image streams inside a single transformer.

Where CLIP and T5 weight the *output* of the text encoder, the emphasis here
weights the *input* embeddings instead, as Forge Neo's own engine used to, so
a negative weight never scales the representation of a word: it hands the
language model a negated embedding, which stands for no word at all.  Sign
flipping that output cannot recover the concept, which is why negative weights
read as doing nothing here however they were masked afterwards.  The word is
therefore encoded at its magnitude, so the conditioning stays a faithful
representation of it, and the sign is carried separately as a mask that
negates the value projection of those tokens in every single-stream block.
Attention still routes to the concept, and the image stream subtracts it
instead of adding it, which is what NegPiP is.  The SD implementation has the
same shape: it encodes the negated concept with a positive weight for exactly
this reason.

The text engine, a port of ComfyUI's since Forge Neo rewrote them all, does
not weight the prompt at all.  It tokenizes the templated prompt with weights
disabled, so `(foo:-1.0)` reaches the language model as literal characters,
which spell out the very concept that was meant to be removed.  Nor can the
weighting of the encoder it wraps be borrowed: that indexes the output by
token on the axis where Krea 2, tapping twelve layers at once, keeps the
layers.  The hooks below parse the weights themselves and tokenize every
region on its own, between a single copy of the chat template, so the weight
lands on the words alone.  The emphasis is applied to the input embeddings
once the vision tower has expanded every reference image into the embeddings
it stands for, and the engine is left to encode, strip the template and
reshape exactly as it would.
"""

from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

import torch

from backend.args import dynamic_args
from backend.sampling import condition, sampling_function
from modules import shared

ENGINE_METHODS: tuple[str, ...] = ("__call__", "_tokenize_with_weights")
ENGINE_REQUIREMENTS: tuple[str, ...] = (
    "_tokenize_with_weights",
    "text_encoder.process_tokens",
    "tokenizer.tokenizer",
)

IMAGE_PAD: int = 151655
"""`<|image_pad|>`, which the engine swaps for the image it stands for"""


class _EngineState:
    """Carries the per-prompt data between the patched text engine methods."""

    engine_cls: type = None
    """the patched text engine class"""

    encoder = None
    """the text encoder whose `process_tokens` is hooked"""

    emphasis = None
    """emphasis of the engine call in progress, as configured by the user"""

    pending: tuple[str, list[float]] = None
    """the prompt being encoded, and the weights of its tokens"""

    aligned: dict[str, list[float]] = None
    """^ the same weights per prompt, expanded over the embeddings of every image"""

    masks: list[torch.Tensor] = None
    """sign masks of the last engine call, one per prompt"""

    dit_cls: type = None
    """the patched transformer class"""

    hooked: set = None
    """the transformers whose value projections are hooked"""

    reported: bool = False
    """whether this generation has confirmed the mask reaching attention"""

    def reset(self):
        self.emphasis = None
        self.pending = None
        self.aligned = None
        self.masks = None
        self.hooked = set()
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

    # Validate the model before changing any global methods.  The value
    # projections are hooked later, from inside the transformer's own forward,
    # but a model without them should still fail here rather than mid sampling.
    engine = _find_text_engine(model)
    _value_projections(dit)
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
    _hook_value_projections(dit, True)
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
    """Encode every prompt as the engine does, keeping its emphasis weights."""

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
        return zs

    _state.emphasis = option
    _state.aligned = {}
    try:
        zs = _originals["__call__"](self, texts, images)
        aligned = _state.aligned
    finally:
        _state.emphasis = None
        _state.pending = None
        _state.aligned = None

    # the engine records "None", which is how it would have read them itself
    if emphasized:
        dynamic_args.last_extra_generation_params["Emphasis"] = option.name

    masks: list[torch.Tensor] = []
    for line, z in zip(texts, zs):
        weights = aligned.get(line, None)
        # a silent no-op is the very failure this module exists to avoid
        if weights is None:
            raise RuntimeError("NegPiP could not align Krea 2's emphasis weights")
        masks.append(_sign_mask(weights, z))

    _state.masks = masks
    return zs


def _negpip_tokenize_with_weights(self, text: str, images: list[torch.Tensor] = []):
    """Tokenize the prompt as one templated conversation, keeping its weights.

    Upstream tokenizes the templated prompt as a whole, with weights disabled.
    Each region is tokenized on its own here instead, between a single copy of
    the template, which keeps the prompt well formed and puts the weight on
    the words alone.  The encoder is handed a weight of 1.0 throughout, since
    it would apply anything else across the layers rather than the tokens, and
    the weights themselves are kept for `process_tokens` to apply.
    """

    if _state.emphasis is None:
        # not called by NegPiP's engine call, so there is nothing to keep
        return _originals["_tokenize_with_weights"](self, text, images)

    from backend.text_processing import parsing

    regions: list[list] = [
        [region, weight]
        for region, weight in parsing.parse_prompt_attention(text, _state.emphasis.name)
        # a chunking hint for CLIP, which parses to a weight of -1 and would
        # otherwise be negated as if it were part of the prompt
        if region != "BREAK"
    ]
    if not regions:
        regions = [["", 1.0]]

    # upstream strips the prompt before templating it
    regions[0][0] = regions[0][0].lstrip()
    regions[-1][0] = regions[-1][0].rstrip()

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
        # the mask and applied to the value projection instead
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
        _state.reported = False
        conds = original(prompt)
        masks, _state.masks = _state.masks, None

        # Krea 2 returns one [tokens, layers, features] tensor per prompt.
        # Keep this deliberately strict: silently losing the mask is worse than
        # exposing an upstream conditioning API change.
        if not isinstance(conds, list) or len(conds) != len(prompt):
            raise RuntimeError("Unexpected Krea 2 conditioning result")
        if masks is None or len(masks) != len(conds):
            raise RuntimeError("Unexpected Krea 2 conditioning masks")

        contexts, negpip_masks = [], []
        count = 0

        for cond, mask in zip(conds, masks):
            if not isinstance(cond, torch.Tensor):
                raise RuntimeError("Unexpected Krea 2 conditioning item")

            mask = mask.to(cond)
            count += int((mask < 0).sum())
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


def _hook_dit_forward(dit, remove: bool):
    """Patch the transformer class rather than the instance we were handed.

    `forge_objects` is rebuilt from `forge_objects_after_applying_lora` before
    every sampling, and the patcher restores its object patches around it, so
    an attribute set on one instance at `process_batch` time is not reliably
    the attribute that runs.  Patching the class always runs, and the value
    projections are then hooked on whichever instance actually arrives.
    """

    if remove:
        cls = _state.dit_cls
        original = _dit_originals.pop("forward", None)
        if cls is not None and original is not None:
            if getattr(cls.forward, "_negpip_krea2", False):
                cls.forward = original
        for hooked in _state.hooked:
            _hook_value_projections(hooked, True)
        _state.hooked.clear()
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
        if mask is not None and self not in _state.hooked:
            _hook_value_projections(self, False)
            _state.hooked.add(self)

        self._negpip_mask = mask
        try:
            return original(self, *args, **kwargs)
        finally:
            self._negpip_mask = None

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


def _single_blocks(dit):
    blocks = None
    for name in ("blocks", "single_blocks", "transformer_blocks"):
        blocks = getattr(dit, name, None)
        if blocks is not None:
            break
    if blocks is None:
        raise RuntimeError("NegPiP could not find Krea 2 single-stream blocks")
    return blocks


def _value_projections(dit) -> list:
    """The value projection of every single-stream block."""

    projections = []
    for block in _single_blocks(dit):
        attention = getattr(block, "attn", getattr(block, "attention", None))
        projection = getattr(attention, "wv", None)
        if projection is None:
            raise RuntimeError("NegPiP could not find a Krea 2 value projection")
        projections.append(projection)

    if not projections:
        raise RuntimeError("NegPiP found no Krea 2 single-stream blocks")

    return projections


def _hook_value_projections(dit, remove: bool):
    for projection in _value_projections(dit):
        if remove:
            original = getattr(projection, "negpip_orig_forward", None)
            if original is not None:
                if getattr(projection.forward, "_negpip_krea2", False):
                    projection.forward = original
                del projection.negpip_orig_forward
            continue

        if getattr(projection.forward, "_negpip_krea2", False):
            continue  # installed by an earlier forward of the same transformer

        original = projection.forward
        projection.negpip_orig_forward = original

        @torch.inference_mode()
        @wraps(original)
        def forward(x, *args, __original=original, **kwargs):
            values = __original(x, *args, **kwargs)
            mask = getattr(dit, "_negpip_mask", None)
            if mask is None:
                return values
            if values.shape[0] % mask.shape[0]:
                raise RuntimeError("Krea 2 NegPiP mask batch does not match attention")
            if values.shape[0] != mask.shape[0]:
                mask = mask.repeat(values.shape[0] // mask.shape[0], 1, 1)
            # the prompt is always at the front of the reference and image streams
            text_length = min(values.shape[1], mask.shape[1])
            values[:, :text_length] *= mask[:, :text_length].to(values)

            if not _state.reported:
                _state.reported = True
                negated = int((mask[:, :text_length] < 0).sum())
                print(
                    f"NegPiP Applied (Krea 2: {negated} of {text_length} text "
                    f"tokens, {values.shape[1]} in the stream)"
                )

            return values

        forward._negpip_krea2 = True
        projection.forward = forward
