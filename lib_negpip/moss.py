"""NegPiP support for MOSS-Image on Forge Neo.

MOSS-Image reaches Forge Neo two ways: Forge's own support (2026-10-21), and the
moss-image-forge-neo extension, which takes the model over while it is installed.
The model is the same either way.  Qwen3.5 encodes the whole chat-templated
prompt, two text-only refiner blocks read it, and 42 single-stream blocks then
attend over the image tokens followed by the text tokens.  Both backends also
edit: reference images run as a branch of their own, whose keys and values
every block attends to after the prompt (or, on some of Forge's reference
methods, as tokens after the prompt), and Forge's `ai-toolkit` method also shows
the pictures to Qwen3.5, ahead of the prompt.

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

The text engine of each backend is hooked rather than the model's own
`get_learned_conditioning`, which is what sets up and clears the reference
images of an edit, and has to keep doing so.  The mask reaches the blocks
through the override Forge's attention function takes in `transformer_options`,
which every attention backend honours, so nothing inside the blocks is
reimplemented and no backend is forced.  The two refiners are left alone: they
attend over the prompt alone, and the concept has to survive them intact for
attention to route to it.  So is the reference branch, which never sees the
prompt.
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

from backend import attention
from backend.args import dynamic_args
from backend.sampling import condition, sampling_function
from backend.text_processing import emphasis, parsing
from modules import shared
from modules.prompt_parser import get_learned_conditioning_prompt_schedules

MASK_KEY: str = "c_negpip_mask"
OVERRIDE_KEY: str = "optimized_attention_override"

IM_START: str = "<|im_start|>"
"""a prompt that already is chat markup, which this Extension leaves alone"""

STRIPPED_LAYOUT = re.compile(r'\s*"(summary|items)"\s*:')
"""a layout whose braces sd-dynamic-prompts stripped, which reads `{x}` as a variant"""

CAPTION: str = '"caption"'
"""the key of the scene's caption and of every item's, the prose of a layout"""

PLACEHOLDER: int = 0xF0000
"""private use code points that stand in for JSON escapes while a caption is parsed"""

PROMPT_FIELDS: tuple[str, ...] = (
    "prompts",
    "negative_prompts",
    "hr_prompts",
    "hr_negative_prompts",
)


class _State:
    """Carries what the hooks share between them."""

    backend = None
    """the backend of the patched model"""

    engine_cls: type = None
    """the patched text engine class"""

    dit_cls: type = None
    """the patched transformer class"""

    masks: Optional[list[torch.Tensor]] = None
    """the masks of the conditioning being encoded, one per prompt; None otherwise"""

    references: int = 0
    """how deep the transformer is in its reference branch, which never sees the prompt"""

    reported: bool = False
    """whether this generation has confirmed the mask reaching attention"""

    def reset(self):
        self.backend = None
        self.masks = None
        self.references = 0
        self.reported = False


_state = _State()
_engine_originals: dict[str, Callable] = {}
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
        _state.backend = backend
        _hook_engine(backend.engine, False)
        _hook_get_learned_conditioning(model, False)
        _hook_dit(dit, False)
        _hook_compile_conditions(False)
    except Exception:
        # Each remover tolerates a partially installed hook set.
        _unhook(model)
        del cls._moss_patched_model
        raise
    cls._patched[3] = True


def _unhook(model):
    _hook_get_learned_conditioning(model, True)
    _hook_engine(None, True)
    _hook_dit(None, True)
    _hook_compile_conditions(True)
    _state.reset()


def has_weights(p: "StableDiffusionProcessing") -> bool:
    """Whether any prompt of the batch carries a weight for this Extension to read.

    Any weight, not only a negative one: the engine reads them all as literal
    text, so one this Extension left alone would reach the model with its
    brackets on.  The prompts are scheduled first, as Forge does before the
    engine sees them, so that `[foo:bar:0.5]` is not mistaken for de-emphasis,
    and the structure of a layout prompt is data rather than a weight.
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

    verbatim: bool = False
    """whether the engine hands the prompt over as typed; this one unescapes `\\(`"""

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

    def ids(self, text: str) -> list[int]:
        # the very call `tokenize_with_weights` makes, which adds no special tokens
        return list(self.engine.tokenizer.tokenizer(text)["input_ids"])

    def head(self, images: list) -> list:
        """The tokens ahead of the prompt: the template, and the pictures of an edit.

        With pictures, the engine is asked to tokenize an empty prompt, so that
        however it introduces them, they come out exactly as it does, the image
        tokens already swapped for the pictures they stand for.
        """

        prefix, _, suffix = self.template.partition("{}")
        if not images:
            return self.ids(prefix)

        tokenize = getattr(self.engine, "_tokenize_with_weights", None)
        if not callable(tokenize):
            raise RuntimeError("NegPiP could not find MOSS-Image's reference prompt")
        chunks = tokenize("", images)
        tokens = [pair[0] for pair in chunks[0]]
        tail = self.ids(suffix) if suffix else []
        if len(chunks) != 1 or tokens[len(tokens) - len(tail) :] != tail:
            raise RuntimeError("NegPiP could not find MOSS-Image's reference prompt")
        return tokens[: len(tokens) - len(tail)]

    def encode_plain(self, line: str, images: list) -> torch.Tensor:
        # the engine's own call, for one line
        return _engine_originals["__call__"](self.engine, [line], images)[0]


class _Extension:
    """The moss-image-forge-neo extension: `MossImageEngine`, with `MossTextEngine`."""

    verbatim: bool = True
    """whether the engine hands the prompt over as typed; this one does"""

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

    def ids(self, text: str) -> list[int]:
        tokenizer = self.engine.tokenizer.tokenizer
        return list(tokenizer.encode(text, add_special_tokens=False))

    def head(self, images: list) -> list:
        # the extension shows the text encoder no pictures; its edit is the DiT's
        return self.ids(self.template.partition("{}")[0])

    def encode_plain(self, line: str, images: list) -> torch.Tensor:
        # what the engine's own call does for each line
        return self.engine.encode(line)


def _backend(model):
    for backend in (_Native, _Extension):
        if backend.matches(model):
            return backend(model)

    raise RuntimeError(
        f"NegPiP does not know this MOSS-Image engine ({type(model).__name__})"
    )


# ================================================================================ #
# Prompt


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


def _json_strings(text: str, end: int) -> list[tuple[int, int]]:
    """`(start, stop)` of every JSON string in `text[:end]`, quotes included."""

    spans = []
    i = 0
    while i < end:
        if text[i] != '"':
            i += 1
            continue
        j = i + 1
        while j < end and text[j] != '"':
            j += 2 if text[j] == "\\" else 1
        spans.append((i, min(j + 1, end)))
        i = j + 1
    return spans


def _caption_spans(text: str, end: int) -> list[tuple[int, int]]:
    """`(start, stop)` of the content of every caption, quotes excluded.

    The scene's caption and every item's are what the layout says in words;
    everything else, the words to render, the meta, the labels and the boxes,
    is data.
    """

    strings = _json_strings(text, end)
    spans = []
    for (key_start, key_stop), (value_start, value_stop) in zip(strings, strings[1:]):
        if text[key_start:key_stop] != CAPTION:
            continue
        if text[key_stop:value_start].strip() != ":":
            continue
        spans.append((value_start + 1, value_stop - 1))
    return spans


def _protect(text: str) -> tuple[str, list[str]]:
    """The JSON escapes of a caption, swapped for characters the parser leaves alone.

    The parser reads a backslash as its own escape, so `\\"` would lose its
    backslash, and `\\\\` half of it.  `\\(` and `\\)` stay, as the escapes of
    Forge that they are: parentheses meant literally.
    """

    out: list[str] = []
    table: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and text[i + 1 : i + 2] not in ("(", ")", ""):
            size = 6 if text[i + 1] == "u" else 2
            out.append(chr(PLACEHOLDER + len(table)))
            table.append(text[i : i + size])
            i += size
        else:
            out.append(text[i])
            i += 1
    return "".join(out), table


def _restore(text: str, table: list[str]) -> str:
    if not table:
        return text
    return "".join(
        (
            table[ord(c) - PLACEHOLDER]
            if PLACEHOLDER <= ord(c) < PLACEHOLDER + len(table)
            else c
        )
        for c in text
    )


def _caption(text: str) -> Optional[list[tuple[str, float]]]:
    """A caption as `(text, weight)` pieces, or `None` when it carries no weight.

    The caption is prose, so it is laid out as a sentence is, except that the
    whitespace inside its quotes is kept as it is.
    """

    protected, table = _protect(text)
    segments = _segments(protected)
    if all(weight == 1.0 for _, weight in segments):
        return None

    lead = protected[: len(protected) - len(protected.lstrip())]
    trail = protected[len(protected.rstrip()) :]
    pieces = _join(segments)
    if not pieces:
        return None

    first, weight = pieces[0]
    pieces[0] = (lead + first, weight)
    if trail:
        pieces.append((trail, 1.0))
    return [(_restore(piece, table), weight) for piece, weight in pieces]


def _split_layout(backend, text: str) -> Optional[tuple[list[tuple[str, float]], str]]:
    """`(pieces, rest)` of a layout prompt: the JSON object it opens with, and what follows.

    A layout is data, not a prompt with weights in it.  Its `bbox` arrays would
    read as de-emphasis to the parser, which would strip their brackets, so it
    is handed to the model as the engine hands it, captions aside: the scene's
    and every item's are prose, and are read for weights.  Only what follows
    the layout, such as the JSON Builder's Extra field, is read as a prompt.

    The Builder escapes every parenthesis, and `\\(` is no JSON escape, so the
    object is looked for in the text the way Forge's engine unescapes it, then
    cut from the text as the engine hands it over.  `None` when the line does
    not open with a layout.
    """

    plain, where = _unescape(text)

    def cut(start: int, stop: int) -> str:
        return text[where[start] : where[stop]] if backend.verbatim else plain[start:stop]

    if STRIPPED_LAYOUT.match(plain):
        # the extension rebuilds the braces, Forge's own engine does not; data
        # either way, and parsing its arrays as weights would only break it more
        return [(cut(0, len(plain)), 1.0)], ""
    if not plain.startswith("{"):
        return None

    try:
        layout, end = json.JSONDecoder().raw_decode(plain)
    except ValueError:
        return None
    if not isinstance(layout, dict):
        return None

    pieces: list[tuple[str, float]] = []
    done = 0
    for start, stop in _caption_spans(plain, end):
        # read in the text as typed, where `\\(` still means a literal parenthesis
        caption = _caption(text[where[start] : where[stop]])
        if caption is None:
            continue
        pieces.append((cut(done, start), 1.0))
        pieces.extend(caption)
        done = stop
    pieces.append((cut(done, end), 1.0))

    return pieces, text[where[end] :]


def _parse(backend, line: str) -> Optional[list[tuple[str, float]]]:
    """The line as `(text, weight)` pieces, or `None` when it carries no weight.

    The pieces are laid out as the one sentence they make, see `_join`.  A line
    without a weight is left to the engine whole, so that it encodes exactly as
    it does without this Extension.
    """

    text = backend.normalize(line)
    if text.startswith(IM_START):
        return None

    found = _split_layout(backend, text)
    if found is None:
        segments = _segments(text)
        if all(weight == 1.0 for _, weight in segments):
            return None
        return _join(segments)

    layout, rest = found
    tail = _join(_segments(rest)) if rest.strip() else []
    if all(weight == 1.0 for _, weight in (*layout, *tail)):
        return None
    if not tail:
        return layout

    # what comes between the layout and the words after it, as `_join` lays out
    # the gap between two pieces
    gap = rest[: len(rest) - len(rest.lstrip())]
    seam = []
    newline = max(gap.rfind("\n"), gap.rfind("\r"))
    if newline >= 0:
        seam = [(gap[: newline + 1], 1.0)]
        gap = gap[newline + 1 :]
    first, weight = tail[0]
    return [*layout, *seam, (" " + first if gap else first, weight), *tail[1:]]


def _join(pieces: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Lay the pieces out as the one sentence they are.

    Qwen's tokenizer keeps the space before a word inside the word's token, so
    `red` in a sentence is the single token ` red`.  The whitespace between two
    pieces therefore goes to the front of the second one, and the weight of the
    word lands on that token.  A run of spaces at a seam is a single space, so
    that `( foo : 0.5 )` weighs `foo` rather than an extra blank.  A line break
    is a token of its own, and belongs to neither side, so it is kept apart at
    1.0.  The ends are stripped, as the engine strips the prompt.
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
        joined.append((" " + core if gap else core, weight))

    return joined


# ================================================================================ #
# Text Engine


def _common(a: list, b: list) -> int:
    """How many leading tokens `a` and `b` share."""

    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def _sentence(backend, pieces: list[tuple[str, float]]) -> tuple[list[int], list[float]]:
    """Tokenize the pieces as the one sentence they make, and weigh every token.

    The sentence is tokenized whole, so its tokens are exactly the ones the
    engine reads for it, whatever the seams between the pieces.  Where each
    piece starts is found by tokenizing the sentence up to it: up to a seam
    between two words, that is the sentence's own tokens, cut there.  A token
    that a seam falls inside belongs to both pieces, and takes whichever of
    their weights is further from 1.0, so that a weight is never lost to it.
    """

    merged: list[tuple[str, float]] = []
    for text, weight in pieces:
        if not text:
            continue
        if merged and merged[-1][1] == weight:
            merged[-1] = (merged[-1][0] + text, weight)
        else:
            merged.append((text, weight))

    if not merged:
        return [], []

    sentence = "".join(text for text, _ in merged)
    tokens = backend.ids(sentence)
    weights = [merged[0][1]] * len(tokens)

    offset = 0
    floor = 0
    for (left, before), (_, after) in zip(merged, merged[1:]):
        offset += len(left)
        head = backend.ids(sentence[:offset])
        cut = max(_common(tokens, head), floor)
        if cut >= len(tokens):
            break
        if cut < len(head):
            # the seam falls inside token `cut`
            shared = before if abs(before - 1.0) > abs(after - 1.0) else after
            weights[cut] = shared
            weights[cut + 1 :] = [after] * (len(tokens) - cut - 1)
            floor = cut + 1
        else:
            weights[cut:] = [after] * (len(tokens) - cut)
            floor = cut

    return tokens, weights


def _tokenize(backend, pieces: list[tuple[str, float]], images: list) -> tuple[list, list[float], int]:
    """Tokenize the pieces inside a single copy of the template.

    Returns the tokens, a weight for each, and how many of them come after the
    last picture, which the text encoder expands into the many embeddings each
    picture stands for.  The template is emitted once around the whole
    sentence, as the engine emits it.
    """

    _, _, suffix = backend.template.partition("{}")
    head = backend.head(images)
    text, weights = _sentence(backend, pieces)
    tail = backend.ids(suffix) if suffix else []

    tokens = [*head, *text, *tail]
    weights = [*([1.0] * len(head)), *weights, *([1.0] * len(tail))]

    if backend.max_length is not None:
        # the engine truncates its own stream, so the weights are cut with it
        tokens = tokens[: backend.max_length]
        weights = weights[: backend.max_length]

    pictures = [i for i, token in enumerate(tokens) if isinstance(token, dict)]
    after = len(tokens) - (pictures[-1] + 1 if pictures else 0)
    return tokens, weights, after


def _encode_line(backend, line: str, images: list, ignore: bool):
    """One prompt line -> `(conditioning, mask, parsed)`, the tokens first."""

    pieces = _parse(backend, line)
    if pieces is None:
        z = backend.encode_plain(line, images)
        return z, z.new_ones(z.shape[0], 1), False

    tokens, weights, after = _tokenize(backend, pieces, images)
    if ignore:
        # "Ignore" parses the weights and declines to apply them, which leaves
        # NegPiP the sign, as on every other model
        weights = [weight if weight < 0.0 else 1.0 for weight in weights]

    pairs = [[(token, 1.0) for token in tokens]]
    z = backend.engine.text_encoder.encode_token_weights(pairs)[0][0]

    # a silent misplacement is the very failure this module exists to avoid;
    # every weight sits after the pictures, which the encoder expands into
    # embeddings of their own, so with pictures the mask is aligned from the end
    if after == len(tokens):
        aligned = z.shape[0] == len(tokens)
    else:
        aligned = z.shape[0] >= after
    if not aligned:
        raise RuntimeError("NegPiP could not align MOSS-Image's prompt weights")

    mask = z.new_ones(z.shape[0], 1)
    if after:
        mask[z.shape[0] - after :, 0] = torch.tensor(
            weights[len(weights) - after :], device=z.device, dtype=z.dtype
        )
    return z, mask, True


def _hook_engine(engine, remove: bool):
    """Patch the text engine class, as `__call__` bypasses the instance."""

    if remove:
        for name, original in _engine_originals.items():
            if _state.engine_cls is not None:
                current = getattr(_state.engine_cls, name, None)
                if getattr(current, "_negpip_moss", False):
                    setattr(_state.engine_cls, name, original)
        _engine_originals.clear()
        _state.engine_cls = None
        return

    engine_cls = type(engine)
    if getattr(engine_cls.__call__, "_negpip_moss", False):
        # its original is gone with whatever installed it, and calling into
        # this hook from itself would never return
        raise RuntimeError("NegPiP is already hooked into MOSS-Image's text engine")
    _state.engine_cls = engine_cls
    _engine_originals["__call__"] = engine_cls.__call__
    engine_cls.__call__ = _negpip_call


def _negpip_call(self, texts: list[str], images: list = []):
    """Encode every prompt as the engine does, keeping a mask of its weights."""

    original = _engine_originals["__call__"]
    backend = _state.backend
    collecting = _state.masks

    if collecting is None or backend is None or self is not backend.engine:
        # not a conditioning NegPiP is collecting
        return original(self, texts, images) if images else original(self, texts)

    name = emphasis.get_current_option(shared.opts.emphasis).name
    if name == "None":
        # which reads a weight as literal characters, exactly as the engine does
        zs = original(self, texts, images) if images else original(self, texts)
        collecting.extend(z.new_ones(z.shape[0], 1) for z in zs)
        return zs

    # as the engine records it, for prompts this Extension leaves alone
    if any(emphasis.uses_emphasis(text) for text in texts):
        dynamic_args.last_extra_generation_params["Emphasis"] = "None"

    zs, masks = [], []
    cache: dict[str, tuple[torch.Tensor, torch.Tensor, bool]] = {}
    parsed = False

    for line in texts:
        if line not in cache:
            cache[line] = _encode_line(backend, line, list(images), name == "Ignore")
        z, mask, read = cache[line]
        zs.append(z)
        masks.append(mask)
        parsed = parsed or read

    if parsed:
        # the engine records "None", which is not what was applied
        dynamic_args.last_extra_generation_params["Emphasis"] = name

    collecting.extend(masks)
    return zs


_negpip_call._negpip_moss = True


def _hook_get_learned_conditioning(model, remove: bool):
    """Collect the masks of the conditioning the model encodes.

    The model's own method stays in charge: it is what hands the pictures of an
    edit to the engine and the transformer, and what clears them again when
    there are none, so that an edit cannot leak into the next generation.
    """

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
        _state.masks = []
        _state.reported = False
        try:
            conds = original(prompt)
            masks = _state.masks
        finally:
            _state.masks = None

        # Keep this deliberately strict: silently losing the mask is worse than
        # exposing an upstream conditioning API change.
        if not isinstance(conds, list) or len(masks) != len(conds):
            raise RuntimeError("NegPiP could not align MOSS-Image's prompt weights")

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

        # lists, so that prompts of different lengths are batched apart downstream
        return {"crossattn": conds, MASK_KEY: masks}

    get_learned_conditioning._negpip_moss = True
    model.get_learned_conditioning = get_learned_conditioning


# ================================================================================ #
# Conditioning


def _hook_compile_conditions(remove: bool):
    """Teach `compile_conditions` the shape the text hook returns.

    It knows a bare tensor, or a dict with both `crossattn` and a pooled
    `vector`, which MOSS-Image has none of.  The mask rides as the conditioning
    itself rides, a `ConditionCrossAttn`, so that it is batched exactly as the
    conditioning is: together when the prompts are the same length, apart when
    they are not, each row of it staying on its own token.
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


def _hook_dit(dit, remove: bool):
    """Patch the transformer class rather than the instance we were handed.

    `forge_objects` is rebuilt from `forge_objects_after_applying_lora` before
    every sampling, so the class is what reliably runs.  The mask arrives as a
    keyword of the forward, which both transformers would otherwise discard,
    and is turned into the attention override of that one call.

    The extension computes the keys and values of its reference pictures from
    inside the forward, with the same options, so that branch is fenced off:
    it never sees the prompt, and what it computes is kept for every step.
    """

    if remove:
        cls = _state.dit_cls
        for name, original in _dit_originals.items():
            if cls is not None and getattr(getattr(cls, name, None), "_negpip_moss", False):
                setattr(cls, name, original)
        _dit_originals.clear()
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

        x = args[0] if args else kwargs.get("x", None)
        context = kwargs.get("context", args[2] if len(args) > 2 else None)
        if not torch.is_tensor(x) or x.ndim != 4:
            raise RuntimeError("MOSS-Image NegPiP could not find the image tokens")
        if torch.is_tensor(context) and context.shape[1] != mask.shape[1]:
            raise RuntimeError("MOSS-Image NegPiP mask does not match the prompt")

        args = list(args)
        if len(args) > 3:
            options = dict(args[3] or {})
            args[3] = options
        else:
            options = dict(kwargs.get("transformer_options", None) or {})
            kwargs["transformer_options"] = options

        # both transformers embed every latent pixel as a token, ahead of the prompt
        image = x.shape[-2] * x.shape[-1]
        previous = options.get(OVERRIDE_KEY, None)
        options[OVERRIDE_KEY] = _attention_override(mask, image, previous)
        return original(self, *args, **kwargs)

    forward._negpip_moss = True
    cls.forward = forward

    if callable(getattr(cls, "reference_kv", None)):
        reference_original = cls.reference_kv
        _dit_originals["reference_kv"] = reference_original

        @wraps(reference_original)
        def reference_kv(self, *args, **kwargs):
            _state.references += 1
            try:
                return reference_original(self, *args, **kwargs)
            finally:
                _state.references -= 1

        reference_kv._negpip_moss = True
        cls.reference_kv = reference_kv


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


def _attention_override(mask: torch.Tensor, image: int, previous: Optional[Callable]):
    """The attention override that weighs the values of the weighted words.

    An override already installed by someone else is chained, so that both
    apply.  Forge's own transformer hands the attention `[batch, tokens,
    heads * dim]`; the extension's hands it `[batch, heads, tokens, dim]`, and
    says so with `skip_reshape`.
    """

    def override(func, *args, **kwargs):
        attend = func if previous is None else partial(previous, func)

        if (
            not _state.references
            and len(args) >= 4
            and torch.is_tensor(args[0])
            and torch.is_tensor(args[2])
        ):
            heads_first = bool(kwargs.get("skip_reshape", False))
            values = _weigh_values(args[0], args[2], mask, image, heads_first)
            if values is not None:
                args = (*args[:2], values, *args[3:])

        return attend(*args, **kwargs)

    return override


def _weigh_values(
    q: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    image: int,
    heads_first: bool,
) -> Optional[torch.Tensor]:
    """Scale the values of the weighted text tokens about the prompt's mean.

    The prompt follows the image tokens in the joint stream of both
    transformers, one for one with the mask.  Whatever an edit adds comes after
    it: the pictures as tokens of their own, or their keys and values, joined to
    the prompt's.  A stream that holds less than the image and the prompt is a
    text refiner, which is left alone.

    The mean is taken over the tokens the mask leaves alone, so that a weighted
    word cannot drag its own reference towards itself.  Everything is decided
    on the device: reading a flag back to the host here would stall the stream
    in every block on every step.
    """

    axis = 2 if heads_first else 1
    if v.ndim != axis + 2 or q.ndim != axis + 2:
        return None

    length, text = v.shape[axis], mask.shape[1]
    end = image + text
    if length < end or q.shape[axis] < end:
        return None

    m = mask
    if m.shape[0] != v.shape[0]:
        if v.shape[0] % m.shape[0]:
            raise RuntimeError("MOSS-Image NegPiP mask batch does not match attention")
        m = m.repeat(v.shape[0] // m.shape[0], 1, 1)
    m = m.to(device=v.device, dtype=torch.float32)
    if heads_first:
        m = m.unsqueeze(1)

    values = v.narrow(axis, image, text).float()
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

    parts = [v.narrow(axis, 0, image), weighted]
    if length > end:
        parts.append(v.narrow(axis, end, length - end))
    return torch.cat(parts, dim=axis)
