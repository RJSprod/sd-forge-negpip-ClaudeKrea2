import importlib
import importlib.util
import os
import re
import sys
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

import torch

ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY: str = os.path.join(ROOT, "lib_negpip")

# every Extension shares one module namespace, so the package is imported under
# a name that belongs to this folder alone
PACKAGE: str = "lib_negpip_" + re.sub(r"\W", "_", os.path.basename(ROOT))


def _import_library():
    """Import this Extension's own `lib_negpip`, by path.

    `lib_negpip` is a name generic enough to collide, and Forge loads every
    Extension into one namespace: another copy of NegPiP installed alongside
    this one imports first, `sys.modules["lib_negpip"]` becomes theirs, and
    `lib_negpip.krea2` is then missing however plainly the file is sitting in
    this folder.  Upstream has no `krea2` module at all, so a stock NegPiP next
    to this one takes Krea 2 support down with it.

    Loading by path under a name of our own cannot be shadowed, and leaves
    whatever is under `lib_negpip` alone for whoever put it there.
    """

    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE,
            os.path.join(LIBRARY, "__init__.py"),
            submodule_search_locations=[LIBRARY],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[PACKAGE]
            raise

    other = sys.modules.get("lib_negpip", None)
    other_path = os.path.dirname(getattr(other, "__file__", "") or "")
    if other_path and os.path.normcase(other_path) != os.path.normcase(LIBRARY):
        print(
            "NegPiP: another copy of the Extension is installed at "
            f"{os.path.dirname(other_path)} ; only one of them can be active"
        )

    return sys.modules[PACKAGE]


_library = _import_library()
_utils = importlib.import_module(f"{PACKAGE}.utils")

IS_NEO: bool = _library.IS_NEO
INCOMPATIBLE_EXTENSIONS: set[str] = _library.INCOMPATIBLE_EXTENSIONS

NEG_PATTERN = _utils.NEG_PATTERN
any_negative = _utils.any_negative
hr_dealer = _utils.hr_dealer
is_krea2 = _utils.is_krea2
reset_prompt_cache = _utils.reset_prompt_cache

from modules import scripts
from modules.prompt_parser import (
    SdConditioning,
    get_learned_conditioning,
    get_learned_conditioning_prompt_schedules,
)
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser


def _support(module: str, function: str) -> Optional[Callable]:
    """Import the support for one model family.

    Importing these at the top of the file means a single missing or
    incompatible module raises out of the Extension, and Forge then skips it
    entirely: SD, Anima and Krea 2 all go down together over any one of them,
    and the only sign is a traceback at startup that scrolls away.  A family
    that cannot be imported is disabled on its own instead, and says so at the
    point where it was actually wanted.
    """

    try:
        return getattr(importlib.import_module(f"{PACKAGE}.{module}"), function)
    except Exception as error:
        print(f"NegPiP: no {module} support ({type(error).__name__}: {error})")
        return None


patch_anima_negpip = _support("anima", "patch_anima_negpip")
patch_krea2_negpip = _support("krea2", "patch_krea2_negpip")
patch_sd_negpip = _support("sd", "patch_sd_negpip")

SUPPORTED: dict[str, Optional[Callable]] = {
    "SD": patch_sd_negpip,
    "Anima": patch_anima_negpip,
    "Krea 2": patch_krea2_negpip,
}


def _verify_ext(p: " StableDiffusionProcessing"):
    for ext in p.scripts.scripts:
        if ext.title() not in INCOMPATIBLE_EXTENSIONS:
            continue
        if p.script_args[ext.args_from] is True:
            return False

    return True


class NegPiP(scripts.Script):
    _patched: list[bool] = [False, False, False]

    _announced: bool = False
    """whether this session has confirmed the Extension is being called"""

    def __init__(self):
        self.active: bool = False

        self.is_xl: bool
        self.is_anima: bool
        self.is_krea2: bool
        self.is_hr: bool

        self.tokenizer: torch.nn.Module

        self.has_hr_p: bool
        self.has_hr_n: bool
        self.rev: bool
        self.batch_size: int

        self.conds: list[torch.Tensor]
        self.c_len: int
        self.c_tokens: list[int]
        self.conds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]
        self.hr_conds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]

        self.unconds: list[torch.Tensor]
        self.uc_len: int
        self.uc_tokens: list[int]
        self.unconds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]
        self.hr_unconds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]

        on_cfg_denoiser(self.denoiser_callback)

    def reset(self):
        self.active = False

        self.is_xl = False
        self.is_anima = False
        self.is_krea2 = False
        self.is_hr = False

        self.tokenizer = None

        self.conds = None
        self.c_tokens = None
        self.conds_all = None
        self.hr_conds_all = None

        self.unconds = None
        self.uc_tokens = None
        self.unconds_all = None
        self.hr_unconds_all = None

        if patch_sd_negpip is not None:
            patch_sd_negpip(None, NegPiP, unpatch=True)
        if patch_anima_negpip is not None:
            patch_anima_negpip(NegPiP, unpatch=True)
        if patch_krea2_negpip is not None:
            patch_krea2_negpip(NegPiP, unpatch=True)

    def title(self):
        return "NegPiP"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    def process_batch(self, p: "StableDiffusionProcessing", *args, **kwargs):
        self.reset()

        if not NegPiP._announced:
            # once per session, so an Extension that is loaded but never
            # reaches a prompt can be told apart from one that is not there
            NegPiP._announced = True
            families = [name for name, fn in SUPPORTED.items() if fn is not None]
            print(
                f"NegPiP Loaded ({'Neo' if IS_NEO else 'Classic'}: "
                f"{', '.join(families) if families else 'nothing'})"
            )

        if not any_negative(p):
            return

        if not _verify_ext(p):
            print("NegPiP Disabled")
            return

        model_name = type(p.sd_model).__name__

        if IS_NEO and not p.sd_model.is_webui_legacy_model():
            self.is_anima = model_name == "Anima"
            self.is_krea2 = is_krea2(p.sd_model)

            if self.is_anima:
                if patch_anima_negpip is None:
                    print("NegPiP Disabled (Anima support failed to import)")
                    return
                patch_anima_negpip(NegPiP)
            elif self.is_krea2:
                if patch_krea2_negpip is None:
                    print("NegPiP Disabled (Krea 2 support failed to import)")
                    return
                patch_krea2_negpip(NegPiP, p.sd_model)
            else:
                # the prompt asked for NegPiP, so say why it is not happening;
                # returning quietly reads as the Extension not being installed
                print(f"NegPiP Disabled (unsupported model: {model_name})")
                return

            reset_prompt_cache(p)
            p.extra_generation_params["NegPiP"] = True
            self.active = True
            print(f"NegPiP Active ({model_name})")
            return

        self.is_xl = p.sd_model.is_sdxl
        self.batch_size = p.batch_size
        self.has_hr_p, self.has_hr_n = hr_dealer(p)
        self.rev = p.sampler_name not in ("DDIM", "PLMS", "UniPC")

        if IS_NEO:
            self.tokenizer = (
                p.sd_model.text_processing_engine_l.tokenize_line
                if self.is_xl
                else p.sd_model.text_processing_engine.tokenize_line
            )
        else:
            self.tokenizer = (
                p.sd_model.conditioner.embedders[0].tokenize_line
                if self.is_xl
                else p.sd_model.cond_stage_model.tokenize_line
            )

        nip = self._getScheduledNegPip(p.prompts, p.steps)
        pin = self._getScheduledNegPip(p.negative_prompts, p.steps)

        self.conds_all = self._calc_conds(p, nip)
        self.unconds_all = self._calc_conds(p, pin)

        hr_steps: int = getattr(p, "hr_second_pass_steps", 0) or p.steps

        if self.has_hr_p:
            hr_nip = self._getScheduledNegPip(p.hr_prompts, hr_steps)
            self.hr_conds_all = self._calc_conds(p, hr_nip)

        if self.has_hr_n:
            hr_pin = self._getScheduledNegPip(p.hr_negative_prompts, hr_steps)
            self.hr_unconds_all = self._calc_conds(p, hr_pin)

        def calcChunks(a: int, b: int) -> int:
            return a // b if a % b == 0 else a // b + 1

        if patch_sd_negpip is None:
            print("NegPiP Disabled (SD support failed to import)")
            return

        self.c_len = calcChunks(self.tokenizer(p.prompts[0])[1], 75)
        self.uc_len = calcChunks(self.tokenizer(p.negative_prompts[0])[1], 75)

        patch_sd_negpip(self, NegPiP)
        reset_prompt_cache(p)
        p.extra_generation_params["NegPiP"] = True
        self.active = True

        if len(self.conds_all[0][0][1]) > 0:
            print(f"NegPiP Enable (Positive: {self.conds_all[0][0][1][0][1]})")
        if len(self.unconds_all[0][0][1]) > 0:
            print(f"NegPiP Enable (Negative: {self.unconds_all[0][0][1][0][1]})")

    def postprocess(self, *args, **kwargs):
        self.reset()

    def before_hr(self, *args, **kwargs):
        self.is_hr = True

    def denoiser_callback(self, params: CFGDenoiserParams):
        if (not self.active) or self.is_anima or self.is_krea2:
            return

        conds_list = []
        tokens_list = []

        if self.is_hr and self.has_hr_p:
            conds = self.hr_conds_all
        else:
            conds = self.conds_all

        if conds is not None:
            for step, regions in conds[0]:
                if step >= params.sampling_step + 2:
                    for conds, tokens in regions:
                        conds_list.append(conds)
                        tokens_list.append(tokens)
                    break
            self.conds = conds_list
            self.c_tokens = tokens_list

        unconds_list = []
        uc_tokens_list = []

        if self.is_hr and self.has_hr_n:
            unconds = self.hr_unconds_all
        else:
            unconds = self.unconds_all

        if unconds is not None:
            for step, regions in unconds[0]:
                if step >= params.sampling_step + 2:
                    for unconds, uc_tokens in regions:
                        unconds_list.append(unconds)
                        uc_tokens_list.append(uc_tokens)
                        break
            self.unconds = unconds_list
            self.uc_tokens = uc_tokens_list

    @staticmethod
    def _getScheduledNegPip(
        prompts: list[str], steps: list[int]
    ) -> list[list[tuple[int, list[tuple[str, float]]]]]:
        """extract the prompts with negative weights"""

        output = []

        scheduled = get_learned_conditioning_prompt_schedules(prompts, steps)
        for i, batch_schedule in enumerate(scheduled):
            stepout = []

            for step, prompt in batch_schedule:
                neg_matches: list[str] = re.findall(NEG_PATTERN, prompt)
                neg_targets = []

                for minusmatch in neg_matches:
                    prompts[i] = prompts[i].replace(minusmatch, "")
                    neg_targets.append(minusmatch.strip("(").strip(")"))

                neg_targets: list[tuple[str, str]] = [x.split(":") for x in neg_targets]
                text_weights: list[tuple[str, float]] = []

                for text, weight in neg_targets:
                    if text.strip() in ("BREAK", "AND", "ADDCOL", "ADDROW"):
                        continue
                    if (weight := float(weight)) < 0.0:
                        text_weights.append((text, weight))

                stepout.append((step, text_weights))

            output.append(stepout)

        return output

    def _cond_dealer(
        self, p: "StableDiffusionProcessing", target: tuple[str, float]
    ) -> tuple[torch.Tensor, int]:
        conds = []

        input = SdConditioning(
            [f"({target[0]}:{-target[1]})"],
            width=p.width,
            height=p.height,
        )

        cond = get_learned_conditioning(p.sd_model, input, p.steps)

        _, token_len = self.tokenizer(target[0])

        conds.append(
            cond[0][0].cond[1 : token_len + 2, :]
            if not self.is_xl
            else cond[0][0].cond["crossattn"][1 : token_len + 2, :]
        )

        conds = torch.cat(conds, 0).unsqueeze(0)
        return conds.repeat(self.batch_size, 1, 1), conds.shape[1]

    def _calc_conds(
        self,
        p: "StableDiffusionProcessing",
        targetlist: list[list[tuple[int, list[tuple[str, float]]]]],
    ) -> list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]:
        outconds = []
        for batch in targetlist:
            stepconds = []
            for step, regions in batch:
                regionconds = []
                for targets in regions:
                    conds, c_tokens = self._cond_dealer(p, targets)
                    regionconds.append((conds, c_tokens))
                stepconds.append((step, regionconds))
            outconds.append(stepconds)
        return outconds
