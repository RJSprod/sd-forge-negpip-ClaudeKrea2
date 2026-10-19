import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from . import IS_NEO

NEG_PATTERN = re.compile(r"\(\s*(?:[^\\(:)]|\\[\(\)])+?\s*\:\s*-\s*\d*\.?\d+\s*\)")

KREA2_GUIDANCE: str = "Attention guidance"
"""steer the image away from the negated words, as NAG does"""

KREA2_FLIP: str = "Value flip"
"""negate the attention values of the negated words, as NegPiP does on SD"""

KREA2_METHODS: tuple[str, ...] = (KREA2_GUIDANCE, KREA2_FLIP)

KREA2_DEFAULTS: dict[str, object] = {
    "negpip_krea2_method": KREA2_GUIDANCE,
    "negpip_krea2_nag_tau": 2.5,
    "negpip_krea2_nag_alpha": 0.25,
}


def krea2_settings() -> tuple[str, float, float]:
    """The Krea 2 method and its guidance parameters, as configured.

    Read through the defaults, so that the Extension runs before its settings
    were registered, or on a Forge that does not have them.
    """

    from modules import shared

    opts = getattr(shared, "opts", None)

    def read(key: str):
        value = getattr(opts, key, None)
        return KREA2_DEFAULTS[key] if value is None else value

    method = read("negpip_krea2_method")
    if method not in KREA2_METHODS:
        method = KREA2_DEFAULTS["negpip_krea2_method"]

    return (
        method,
        float(read("negpip_krea2_nag_tau")),
        float(read("negpip_krea2_nag_alpha")),
    )


def is_krea2(model) -> bool:
    """Whether this is Krea 2, without insisting on the name of its class.

    A rename or a repackaged build would otherwise drop straight through to
    doing nothing at all, which is indistinguishable from the Extension not
    being installed.  The module of its text engine is specific to it, where
    `text_processing_engine_qwen` on its own is shared by half a dozen models.
    """

    if type(model).__name__.lower() in ("krea2", "krea_2"):
        return True

    engine = getattr(model, "text_processing_engine_qwen", None)
    if engine is None:
        return False

    return type(engine).__module__.rpartition(".")[2] == "krea2_engine"


def reset_prompt_cache(p: "StableDiffusionProcessing"):
    c = 3 if IS_NEO else 2

    p.cached_c = [None] * c
    p.cached_uc = [None] * c
    if hasattr(p, "cached_hr_c"):
        p.cached_hr_c = [None] * c
        p.cached_hr_uc = [None] * c


def hr_dealer(p: "StableDiffusionProcessing") -> tuple[bool, bool]:
    return (
        bool(getattr(p, "hr_prompts", None)),
        bool(getattr(p, "hr_negative_prompts", None)),
    )


def has_negative(prompt: str) -> bool:
    return bool(re.search(NEG_PATTERN, prompt))


def have_negative(prompts: list[str]) -> bool:
    return any(has_negative(p) for p in prompts)


def any_negative(p: "StableDiffusionProcessing") -> bool:
    return any(
        [
            have_negative(p.prompts),
            have_negative(p.negative_prompts),
            have_negative(getattr(p, "hr_prompts", None) or ""),
            have_negative(getattr(p, "hr_negative_prompts", None) or ""),
        ]
    )
