# SD Forge Negative Prompt in Prompt
This is an Extension for Forge [Classic](https://github.com/Haoming02/sd-webui-forge-classic/tree/classic) / [Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), which implements **NegPip**, allowing you to give words a negative emphasis inside the positive prompt field to suppress a concept, or vice versa.

> [!IMPORTANT]
> Supports **SD1**, **SDXL**, **Anima**, and **Krea 2** on Forge Neo.
> **Krea 2** is supported in **txt2img**, **img2img**, as well as **Edit** *(reference)* mode.
> **Anima** and **Krea 2** need a Forge Neo from 2026-10-01 or later, after its text engines were rewritten.

## How to Use

- add `(foo:-1.0)` in the `positive prompt` to **remove** a concept
- add `(bar:-1.0)` in the `negative prompt` to **enforce** a concept

> [!NOTE]
> **Krea 2** normally reads prompt weights as literal text, so `(foo:-1.0)` would spell out the very concept it was meant to remove. While this Extension is active, the weights are honored, in **Edit** mode too, meaning that regular emphasis *(eg. `(foo:1.5)`)* takes effect as well.

> [!TIP]
> **Krea 2** does not work with the `None` **Emphasis Mode**, which reads a weight as literal characters rather than parsing it. The console says so when that happens; every other mode is fine.

## Krea 2

**Krea 2** is a single-stream transformer run at CFG 1: there is no negative prompt to speak of, and a text token is a tiny share of an attention that also covers the reference and image streams. NegPiP reaches it by one of two methods, chosen under **Settings → NegPiP**:

- **Attention guidance** *(default)*: [Normalized Attention Guidance](https://arxiv.org/abs/2505.21179), the method that restores negative prompting on few-step models. The prompt is encoded without the negated words, the words alone are encoded beside the same reference images, and in every block the image is steered away from them. This is the method that works in **Edit** mode, where the concept to remove is in the reference image itself. It costs one extra attention per block, roughly a third more time per step.
- **Value flip**: what NegPiP does on **SD**, and what every other Krea 2 NegPiP does: the attention values of the negated words are negated, so the image subtracts the concept instead of adding it. Cheaper, and enough for a concept the prompt itself would have drawn; it cannot remove one a reference image shows.

The weight is the dial for both. Under attention guidance `-1.0` is NAG's default guidance scale of 5, and every further `-1.0` adds 4, so `(foo:-2.0)` pushes harder than `(foo:-1.0)`. The normalization threshold *(`tau`, 2.5)* and the blend *(`alpha`, 0.25)* are NAG's defaults and sit in the same settings: raise them for stronger guidance, lower them to keep a reference image more faithfully. Several negated regions are joined into one negative phrase, at the strongest of their weights. The method and its parameters are written to the infotext.

> [!TIP]
> In **Edit** mode, say what should be there instead: `a person in a red shirt, (green shirt:-1.0)` gives the guidance somewhere to go, where `a person in a (green shirt:-1.0)` only takes the words away.

## Console

The Extension reports what it did, so that nothing it decided is silent:

| | |
| --- | --- |
| `NegPiP Loaded` | it is installed and being called, listing the models it can handle |
| `NegPiP Active` | it engaged, naming the model it recognised and, on Krea 2, the method |
| `NegPiP Enable` | how many tokens it is negating |
| `NegPiP Applied` | *(Krea 2)* the method reached the transformer, with its scale and token counts |
| `NegPiP Disabled` | it stood down, and why |

Nothing at all for a prompt that does contain `(foo:-1.0)` means the weight was not recognised as one.

If a model is missing from the `NegPiP Loaded` list, its support could not be imported, and the line above it says why.

> [!WARNING]
> Do not install this Extension alongside the original **sd-webui-negpip**. Both patch the same hooks, and both ship a `lib_negpip` package into the single namespace Forge shares between Extensions. This one says so on the console when it finds another copy.

## Examples

<table>
    <tr>
        <th>Base</th>
        <th><code>(aqua hair:-1.0)</code><br>in <b>Positive</b> Prompt</th>
        <th><code>(aqua hair:1.5)</code><br>in <b>Negative</b> Prompt</th>
    </tr>
    <tr>
        <td><img src="./img/off.webp" width=256></td>
        <td><img src="./img/negpip.webp" width=256></td>
        <td><img src="./img/neg.webp" width=256></td>
    </tr>
</table>

- **Full Prompts**

```
masterpiece, best quality, high quality, 1girl, solo, hatsune miku, vocaloid, casual, looking at viewer, smile, simple background, white background,
anime screenshot, anime coloring, screencap, flat color, masterpiece, best quality, very aesthetic, absurdres, aesthetic, detailed, beautiful color, amazing quality, highres, safe
Negative prompt: (signature), worst quality, bad quality, low quality, text, name, watermark, (hdr, cinematic, high contrast), logo, username, bad anatomy, bad proportions, extra limbs, extra digit, extra legs, extra legs and arms, disfigured, missing arms, too many fingers, fused fingers, missing fingers, unclear eyes, censored
```
