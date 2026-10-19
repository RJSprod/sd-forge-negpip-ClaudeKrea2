import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from . import IS_NEO

NEG_PATTERN = re.compile(r"\(\s*(?:[^\\(:)]|\\[\(\)])+?\s*\:\s*-\s*\d*\.?\d+\s*\)")


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
