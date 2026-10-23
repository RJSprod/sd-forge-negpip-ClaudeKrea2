"""NegPiP support for MOSS-Image on Forge Neo.

MOSS-Image reaches Forge Neo two ways: Forge's own support (2026-10-21), and the
moss-image-forge-neo extension, which takes the model over while it is installed.
The model is the same either way.  Qwen3.5 encodes the whole chat-templated
prompt, two text-only refiner blocks read it, and 42 single-stream blocks then
attend over the image tokens followed by the text tokens.

Neither backend parses prompt weights.  Both hand the prompt to the text encoder
as typed, so `(foo:-1.0)` and `(foo:0.5)` arrive as those very characters,
brackets and number included, and the model is asked for `foo` either way.  On
MOSS-Image this Extension therefore reads every weight, not only a negative one,
with Forge's own parser, as every other model in Forge reads them.

The prompt is encoded as the same sentence without the weights, so the
conditioning is exactly what the model would have read, and the weights travel
beside it as a per-token mask.  In each of the 42 blocks the mask scales the
attention values of the weighted words: attention still routes to them, and
what they contribute to the image is inverted, removed, halved or doubled.  A
weight applied the way SD applies emphasis, by scaling the text embedding, would
all but vanish here, since the refiners and every block normalise what they are
given.

The values are scaled about the mean value of the prompt's other tokens rather
than about zero, `v -> mean + w * (v - mean)`.  The text tokens share a large
common component, and scaling a token's whole value would push or pull that
component as well: a shift that every query in the stream reads, and that every
next block normalises away at the expense of detail.  About the mean, only what
makes the word that word is scaled, so `-1.0` inverts the word without also
subtracting "there is a prompt here" from the whole image.

The mask reaches the blocks through the override Forge's attention function
takes in `transformer_options`, which every attention backend honours, so
nothing inside the blocks is reimplemented and no backend is forced.  The two
refiners are left alone: they attend over the prompt alone, and the concept has
to survive them intact for attention to route to it.
"""

import json
import re
import sys
from functools import partial, wraps
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing
    from scripts.negpip import NegPiP

import torch

from backend import attention, memory_management
from backend.args import dynamic_args
from backend.sampling import condition, sampling_function
from backend.text_processing import emphasis, parsing
from modules import shared
from modules.prompt_parser import get_learned_conditioning_prompt_schedules

MASK_KEY: str = "c_negpip_mask"
OVERRIDE_KEY: str = "optimized_attention_override"

IM_START: str = "<|im_start|>"
"""a prompt that already is chat markup, which the engine templates nothing around"""

STRIPPED_LAYOUT = re.compile(r'\s*"(summary|items)"\s*:')
"""a layout whose braces sd-dynamic-prompts stripped, which reads `{x}` as a variant"""

PROMPT_FIELDS: tuple[str, ...] = (
    "prompts",
    "negative_prompts",
    "hr_prompts",
    "hr_negative_prompts",
)


class _State:
    """Carries what the hooks share between them."""

    dit_cls: type = None
    """the patched transformer class"""

    reported: bool = False
    """whether this generation has confirmed the mask reaching attention"""

    def reset(self):
        self.reported = False


_state = _State()
_dit_originals: dict[str, Callable] = {}


def patch_moss_negpip(cls: "NegPiP", model=None, *, unpatch: bool = False):
    """Install or remove all MOSS-Image hooks.

    The caller passes the model it decided was MOSS-Image, so that the hooks
    cannot land on a different one than the check was made against.
    """
    if unpatch != cls._patched[3]:
        return

    if unpatch:
        model = getattr(cls, "_moss_patched_model", None)
        if model is not None:
            _unhook(model)
            del cls._moss_patched_model
        cls._patched[3] = False
        return

    if model is None:
        model = shared.sd_model

    # Validate everything before changing any global methods
    backend = _backend(model)
    backend.check()
    dit = model.forge_objects.unet.model.diffusion_model
    _check_dit(dit)
    _probe_attention(dit)

    cls._moss_patched_model = model
    try:
        _hook_get_learned_conditioning(model, backend, False)
        _hook_dit_forward(dit, False)
        _hook_compile_conditions(False)
    except Exception:
        # Each remover tolerates a partially installed hook set.
        _unhook(model)
        del cls._moss_patched_model
        raise
    cls._patched[3] = True


def _unhook(model):
    _hook_get_learned_conditioning(model, None, True)
    _hook_dit_forward(None, True)
    _hook_compile_conditions(True)
    _state.reset()


def has_weights(p: "StableDiffusionProcessing") -> bool:
    """Whether any prompt of the batch carries a weight for this Extension to read.

    Any weight, not only a negative one: the engine reads them all as literal
    text, so one this Extension left alone would reach the model with its
    brackets on.  The prompts are scheduled first, as Forge does before the
    engine sees them, so that `[foo:bar:0.5]` is not mistaken for de-emphasis,
    and the layout of a layout prompt is data rather than a weight.
    """

    backend = _backend(p.sd_model)
    steps = max(int(getattr(p, "steps", 1) or 1), 1)

    for field in PROMPT_FIELDS:
        prompts = [x for x in (getattr(p, field, None) or ()) if isinstance(x, str)]
        if not prompts:
            continue
        for schedule in get_learned_conditioning_prompt_schedules(prompts, steps):
            for _, text in schedule:
                if _parse(backend, text) is not None:
                    return True

    return False


# ================================================================================ #
# Backends


def _unescape(text: str) -> tuple[str, list[int]]:
    """`\\(` and `\\)` unescaped, as `SDTokenizer.tokenize_with_weights` does.

    Also returns where each character of the result sits in `text`, plus where
    the result ends, so that a cut made in the result can be made in `text`.
    """

    out: list[str] = []
    where: list[int] = []
    i = 0
    while i < len(text):
        where.append(i)
        if text[i] == "\\" and text[i + 1 : i + 2] in ("(", ")"):
            out.append(text[i + 1])
            i += 2
        else:
            out.append(text[i])
            i += 1
    where.append(len(text))
    return "".join(out), where


class _Native:
    """Forge Neo's own MOSS-Image: `MOSSImage`, with `Qwen359BEngine`."""

    max_length: Optional[int] = None

    def __init__(self, model):
        self.engine = getattr(model, "text_processing_engine_qwen", None)
        self.template = str(getattr(self.engine, "llama_template", ""))

    @staticmethod
    def matches(model) -> bool:
        return type(model).__name__ == "MOSSImage"

    def check(self):
        tokenizer = getattr(self.engine, "tokenizer", None)
        encoder = getattr(self.engine, "text_encoder", None)
        if self.engine is None:
            raise RuntimeError("NegPiP could not find MOSS-Image's text engine")
        if "{}" not in self.template:
            raise RuntimeError("NegPiP could not find MOSS-Image's prompt template")
        if not callable(getattr(tokenizer, "tokenizer", None)):
            raise RuntimeError("NegPiP could not find MOSS-Image's tokenizer")
        if not callable(getattr(encoder, "encode_token_weights", None)):
            raise RuntimeError("NegPiP could not find MOSS-Image's text encoder")

    def normalize(self, line: str) -> str:
        # `Qwen359BEngine.__call__` strips the line, and an empty one is a space
        return line.strip() or " "

    unescape = staticmethod(_unescape)

    def ids(self, text: str) -> list[int]:
        # the very call `tokenize_with_weights` makes, which adds no special tokens
        return list(self.engine.tokenizer.tokenizer(text)["input_ids"])

    def encode_plain(self, line: str) -> torch.Tensor:
        return self.engine([line])[0]


class _Extension:
    """The moss-image-forge-neo extension: `MossImageEngine`, with `MossTextEngine`."""

    def __init__(self, model):
        self.engine = getattr(model, "text_engine_moss", None)
        module = sys.modules.get(type(self.engine).__module__, None)
        self.template = str(getattr(module, "TEMPLATE", ""))
        length = getattr(module, "MAX_LENGTH", None)
        self.max_length = length if isinstance(length, int) else None
        self.restore = getattr(module, "restore_layout_json", None)

    @staticmethod
    def matches(model) -> bool:
        if type(model).__name__ == "MossImageEngine":
            return True
        return getattr(model, "is_moss_image", False) is True

    def check(self):
        tokenizer = getattr(getattr(self.engine, "tokenizer", None), "tokenizer", None)
        encoder = getattr(self.engine, "text_encoder", None)
        if self.engine is None:
            raise RuntimeError("NegPiP could not find MOSS-Image's text engine")
        if "{}" not in self.template:
            raise RuntimeError("NegPiP could not find MOSS-Image's prompt template")
        if not callable(getattr(tokenizer, "encode", None)):
            raise RuntimeError("NegPiP could not find MOSS-Image's tokenizer")
        if not callable(getattr(encoder, "encode_token_weights", None)):
            raise RuntimeError("NegPiP could not find MOSS-Image's text encoder")
        if not callable(getattr(self.engine, "encode", None)):
            raise RuntimeError("NegPiP could not find MOSS-Image's encode")

    def normalize(self, line: str) -> str:
        # `MossTextEngine.token_ids` strips the line, and rebuilds a layout whose
        # braces sd-dynamic-prompts stripped
        text = line.strip()
        if callable(self.restore):
            text = self.restore(text)
        return text

    @staticmethod
    def unescape(text: str) -> tuple[str, list[int]]:
        # the extension hands the prompt to the tokenizer as typed
        return text, list(range(len(text) + 1))

    def ids(self, text: str) -> list[int]:
        tokenizer = self.engine.tokenizer.tokenizer
        return list(tokenizer.encode(text, add_special_tokens=False))

    def encode_plain(self, line: str) -> torch.Tensor:
        return self.engine.encode(line)


def _backend(model):
    for backend in (_Native, _Extension):
        if backend.matches(model):
            return backend(model)

    raise RuntimeError(
        f"NegPiP does not know this MOSS-Image engine ({type(model).__name__})"
    )


# ================================================================================ #
# Text Engine


def _split_layout(backend, text: str) -> tuple[str, str]:
    """`(layout, rest)`: the JSON object a layout prompt opens with, and what follows.

    A layout is data, not a prompt with weights in it.  Its `bbox` arrays would
    read as de-emphasis to the parser, which would strip their brackets, so it
    is handed to the model exactly as the engine hands it, and only what follows
    it, such as the JSON Builder's Extra field, is read for weights.  The
    Builder escapes every parenthesis, and `\\(` is no JSON escape, so the
    object is looked for in the text the way the engine unescapes it.
    """

    plain, where = backend.unescape(text)
    if STRIPPED_LAYOUT.match(plain):
        # the extension rebuilds the braces, Forge's own engine does not; data
        # either way, and parsing its arrays as weights would only break it more
        return plain, ""
    if not plain.startswith("{"):
        return "", text

    try:
        layout, end = json.JSONDecoder().raw_decode(plain)
    except ValueError:
        return "", text
    if not isinstance(layout, dict):
        return "", text

    return plain[:end], text[where[end] :]


def _segments(text: str) -> list[tuple[str, float]]:
    """The prompt as `(text, weight)` pairs, the way Forge's parser reads it.

    `BREAK` is a chunk boundary for CLIP, which this model does not have; the
    engine reads it as the word it is, so it stays one.  The parser swallows
    the whitespace around it, which is put back.
    """

    segments = []
    for segment, weight in parsing.parse_prompt_attention(text):
        if segment == "BREAK" and weight == -1:
            segments.append((" BREAK ", 1.0))
        else:
            segments.append((segment, float(weight)))
    return segments


def _parse(backend, line: str) -> Optional[list[tuple[str, float]]]:
    """The line as `(text, weight)` pieces, or `None` when it carries no weight.

    A line without a weight is left to the engine whole, so that it encodes
    exactly as it does without this Extension.
    """

    text = backend.normalize(line)
    if text.startswith(IM_START):
        return None

    layout, rest = _split_layout(backend, text)
    segments = _segments(rest) if rest.strip() else []
    if all(weight == 1.0 for _, weight in segments):
        return None

    return ([(layout, 1.0)] if layout else []) + segments


def _join(pieces: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Lay the pieces out as the one sentence they are.

    Qwen's tokenizer keeps the space before a word inside the word's token, so
    `red` in a sentence is the single token ` red`.  The whitespace between two
    pieces therefore goes to the front of the second one: the weighted word then
    tokenizes as it does in the sentence, and its weight lands on that token.  A
    run of spaces at a seam is a single space, so that `( foo : 0.5 )` weighs
    `foo` rather than an extra blank.  A line break is a token of its own, and
    belongs to neither side, so it is kept apart at 1.0.  The ends are
    stripped, as the engine strips the prompt.

    A single punctuation mark joins the word after it the same way, so that
    `1girl,(red hair:-1)` is `1girl` and `,red`; the mark goes with the word.
    """

    cores: list[tuple[str, float]] = []
    gaps: list[str] = []
    pending = ""

    for text, weight in pieces:
        core = text.strip()
        if not core:
            pending += text
            continue
        if cores:
            gaps.append(pending + text[: len(text) - len(text.lstrip())])
        cores.append((core, weight))
        pending = text[len(text.rstrip()) :]

    joined = cores[:1]
    for gap, (core, weight) in zip(gaps, cores[1:]):
        newline = max(gap.rfind("\n"), gap.rfind("\r"))
        if newline >= 0:
            joined.append((gap[: newline + 1], 1.0))
            gap = gap[newline + 1 :]

        sentence = "".join(text for text, _ in joined)
        if (
            not gap
            and core[:1].isalpha()
            and sentence[-2:-1].isalnum()
            and not sentence[-1].isalnum()
            and not sentence[-1].isspace()
        ):
            last, left = joined.pop()
            if last[:-1]:
                joined.append((last[:-1], left))
            core = last[-1] + core

        joined.append((" " + core if gap else core, weight))

    return joined


def _tokenize(backend, pieces: list[tuple[str, float]]) -> tuple[list[int], list]:
    """Tokenize the pieces inside a single copy of the template.

    Each piece is tokenized on its own, so that its tokens are its own by
    construction, and the template is emitted once around the whole sentence.
    """

    prefix, _, suffix = backend.template.partition("{}")

    tokens: list[int] = []
    weights: list[float] = []

    for text, weight in ((prefix, 1.0), *_join(pieces), (suffix, 1.0)):
        ids = backend.ids(text) if text else []
        tokens.extend(ids)
        weights.extend([weight] * len(ids))

    if backend.max_length is not None:
        # the engine truncates its own stream, so the weights are cut with it
        tokens = tokens[: backend.max_length]
        weights = weights[: backend.max_length]

    return tokens, weights


def _encode_line(backend, line: str, ignore: bool):
    """One prompt line -> `(conditioning, mask, parsed)`, the tokens first."""

    pieces = _parse(backend, line)
    if pieces is None:
        z = backend.encode_plain(line)
        return z, z.new_ones(z.shape[0], 1), False

    tokens, weights = _tokenize(backend, pieces)
    if ignore:
        # "Ignore" parses the weights and declines to apply them, which leaves
        # NegPiP the sign, as on every other model
        weights = [weight if weight < 0.0 else 1.0 for weight in weights]

    pairs = [[(token, 1.0) for token in tokens]]
    z = backend.engine.text_encoder.encode_token_weights(pairs)[0][0]

    # a silent misplacement is the very failure this module exists to avoid
    if z.shape[0] != len(tokens):
        raise RuntimeError("NegPiP could not align MOSS-Image's prompt weights")

    mask = torch.tensor(weights, device=z.device, dtype=z.dtype).reshape(-1, 1)
    return z, mask, True


def _hook_get_learned_conditioning(model, backend, remove: bool):
    if remove:
        original = getattr(model, "negpip_moss_orig_get_learned_conditioning", None)
        if original is not None:
            if getattr(model.get_learned_conditioning, "_negpip_moss", False):
                # the original is the class method, so the instance attribute goes
                vars(model).pop("get_learned_conditioning", None)
                if model.get_learned_conditioning != original:
                    model.get_learned_conditioning = original
            del model.negpip_moss_orig_get_learned_conditioning
        return

    original = model.get_learned_conditioning
    model.negpip_moss_orig_get_learned_conditioning = original

    @torch.inference_mode()
    @wraps(original)
    def get_learned_conditioning(prompt):
        memory_management.load_model_gpu(model.forge_objects.clip.patcher)

        name = emphasis.get_current_option(shared.opts.emphasis).name
        _state.reported = False

        contexts, masks = [], []
        cache: dict[str, tuple[torch.Tensor, torch.Tensor, bool]] = {}
        parsed = False

        for line in prompt:
            if line not in cache:
                cache[line] = _encode_line(backend, line, name == "Ignore")
            z, mask, read = cache[line]
            contexts.append(z)
            masks.append(mask)
            parsed = parsed or read

        if parsed:
            # the engine records "None", which is not what was applied
            dynamic_args.last_extra_generation_params["Emphasis"] = name

        negated = sum(int((mask < 0.0).sum()) for mask in masks)
        weighted = sum(int(((mask >= 0.0) & (mask != 1.0)).sum()) for mask in masks)

        if negated or weighted:
            key = (
                "Negative"
                if getattr(prompt, "is_negative_prompt", False)
                else "Positive"
            )
            counts = [f"{negated} negated"] if negated else []
            counts += [f"{weighted} weighted"] if weighted else []
            print(f"NegPiP Enable ({key}: {', '.join(counts)})")

        # lists, so that prompts of different lengths are padded downstream
        return {"crossattn": contexts, MASK_KEY: masks}

    get_learned_conditioning._negpip_moss = True
    model.get_learned_conditioning = get_learned_conditioning


# ================================================================================ #
# Conditioning


def _hook_compile_conditions(remove: bool):
    """Teach `compile_conditions` the shape the text hook returns.

    It knows a bare tensor, or a dict with both `crossattn` and a pooled
    `vector`, which MOSS-Image has none of.  The mask rides as the conditioning
    itself rides, a `ConditionCrossAttn`: when the positive and negative prompts
    differ in length, Forge batches them by repeating the shorter one along the
    tokens, and the mask is repeated with them, which keeps every row of it on
    its own token and the batching exactly what it is without this Extension.
    """

    if remove:
        original = getattr(condition, "negpip_moss_orig_compile_conditions", None)
        if original is not None:
            condition.compile_conditions = original
            sampling_function.compile_conditions = original
            del condition.negpip_moss_orig_compile_conditions
        return

    original = condition.compile_conditions
    condition.negpip_moss_orig_compile_conditions = original

    @wraps(original)
    def compile_conditions(cond):
        if isinstance(cond, dict) and "crossattn" in cond and "vector" not in cond:
            crossattn = cond["crossattn"]
            model_conds = {"c_crossattn": condition.ConditionCrossAttn(crossattn)}
            if MASK_KEY in cond:
                model_conds[MASK_KEY] = condition.ConditionCrossAttn(cond[MASK_KEY])
            return [dict(cross_attn=crossattn, model_conds=model_conds)]
        return original(cond)

    compile_conditions._negpip_moss = True
    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions


# ================================================================================ #
# Transformer


def _check_dit(dit):
    """The parts of the transformer the hooks rely on, checked before patching."""

    for name in ("layers", "video_embedder", "text_embedder"):
        if getattr(dit, name, None) is None:
            raise RuntimeError(f"NegPiP could not find MOSS-Image's {name}")
    if getattr(dit.text_embedder, "blocks", None) is None:
        raise RuntimeError("NegPiP could not find MOSS-Image's text refiners")


def _probe_attention(dit):
    """Make sure the attention the transformer calls honours the override.

    The mask has no other way into the blocks, so an attention that ignored
    it would run a generation that quietly does nothing.  This fails here,
    before anything is patched, instead.
    """

    module = sys.modules.get(type(dit).__module__, None)
    function = getattr(module, "attention_function", None)
    if not callable(function):
        function = attention.attention_function

    marker = object()
    probe = torch.zeros(1, 1, 1, 1)
    try:
        seen = function(
            probe,
            probe,
            probe,
            1,
            skip_reshape=True,
            transformer_options={OVERRIDE_KEY: lambda *args, **kwargs: marker},
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
    every sampling, so the class is what reliably runs.  The mask arrives as a
    keyword of the forward, which both transformers would otherwise discard,
    and is turned into the attention override of that one call.
    """

    if remove:
        cls = _state.dit_cls
        original = _dit_originals.pop("forward", None)
        if cls is not None and original is not None:
            if getattr(cls.forward, "_negpip_moss", False):
                cls.forward = original
        _state.dit_cls = None
        return

    cls = type(dit)
    original = cls.forward
    _dit_originals["forward"] = original
    _state.dit_cls = cls

    @wraps(original)
    def forward(self, *args, **kwargs):
        mask = _text_mask(kwargs.pop(MASK_KEY, None))
        if mask is None:
            return original(self, *args, **kwargs)

        context = kwargs.get("context", args[2] if len(args) > 2 else None)
        if torch.is_tensor(context) and context.shape[1] != mask.shape[1]:
            raise RuntimeError("MOSS-Image NegPiP mask does not match the prompt")

        args = list(args)
        if len(args) > 3:
            options = dict(args[3] or {})
            args[3] = options
        else:
            options = dict(kwargs.get("transformer_options", None) or {})
            kwargs["transformer_options"] = options

        previous = options.get(OVERRIDE_KEY, None)
        options[OVERRIDE_KEY] = _attention_override(mask, previous)
        return original(self, *args, **kwargs)

    forward._negpip_moss = True
    cls.forward = forward


def _text_mask(mask) -> Optional[torch.Tensor]:
    """The conditioning mask as `[batch, tokens, 1]`, or `None` if it weights nothing."""

    if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
        return None

    batch, tokens = mask.shape[0], mask.shape[1]
    if mask.numel() != batch * tokens:
        return None

    mask = mask.reshape(batch, tokens, 1)
    if not bool((mask != 1.0).any()):
        return None

    return mask


def _attention_override(mask: torch.Tensor, previous: Optional[Callable]):
    """The attention override that weighs the values of the weighted words.

    An override already installed by someone else is chained, so that both
    apply.  Forge's own transformer hands the attention `[batch, tokens,
    heads * dim]`; the extension's hands it `[batch, heads, tokens, dim]`, and
    says so with `skip_reshape`.
    """

    def override(func, *args, **kwargs):
        attend = func if previous is None else partial(previous, func)

        if len(args) >= 4 and torch.is_tensor(args[2]):
            heads_first = bool(kwargs.get("skip_reshape", False))
            values = _weigh_values(args[2], mask, heads_first)
            if values is not None:
                args = (*args[:2], values, *args[3:])

        return attend(*args, **kwargs)

    return override


def _weigh_values(
    v: torch.Tensor, mask: torch.Tensor, heads_first: bool
) -> Optional[torch.Tensor]:
    """Scale the values of the weighted text tokens about the prompt's mean.

    The text trails the image in the joint stream of both transformers, so the
    last tokens of the stream are the prompt, one for one with the mask.  A
    stream no longer than the prompt is a text refiner, which is left alone.

    The mean is taken over the tokens the mask leaves alone, so that a weighted
    word cannot drag its own reference towards itself.  Everything is decided
    on the device: reading a flag back to the host here would stall the stream
    in every block on every step.
    """

    axis = 2 if heads_first else 1
    if v.ndim != axis + 2:
        return None

    length, text = v.shape[axis], mask.shape[1]
    if length <= text:
        return None

    m = mask
    if m.shape[0] != v.shape[0]:
        if v.shape[0] % m.shape[0]:
            raise RuntimeError("MOSS-Image NegPiP mask batch does not match attention")
        m = m.repeat(v.shape[0] // m.shape[0], 1, 1)
    m = m.to(device=v.device, dtype=torch.float32)
    if heads_first:
        m = m.unsqueeze(1)

    start = length - text
    values = v.narrow(axis, start, text).float()
    keep = (m == 1.0).float()
    mean = (values * keep).sum(dim=axis, keepdim=True)
    mean = mean / keep.sum(dim=axis, keepdim=True).clamp_min(1.0)
    # mean + w * (v - mean), written so that a token at 1.0 is left bit for bit
    weighted = (values + (m - 1.0) * (values - mean)).to(v.dtype)

    if not _state.reported:
        _state.reported = True
        negated = int((mask < 0.0).sum())
        scaled = int(((mask >= 0.0) & (mask != 1.0)).sum())
        print(
            f"NegPiP Applied (MOSS-Image: {negated} negated, {scaled} weighted "
            f"of {text} text tokens, {length} in the stream)"
        )

    return torch.cat((v.narrow(axis, 0, start), weighted), dim=axis)
