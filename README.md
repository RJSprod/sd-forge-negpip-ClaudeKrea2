# SD Forge Negative Prompt in Prompt
This is an Extension for Forge [Classic](https://github.com/Haoming02/sd-webui-forge-classic/tree/classic) / [Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), which implements **NegPip**, allowing you to give words a negative emphasis inside the positive prompt field to suppress a concept, or vice versa.

> [!IMPORTANT]
> Supports **SD1**, **SDXL**, **Anima**, and **Krea 2** on Forge Neo.
> **Krea 2** is supported in **txt2img**, **img2img**, as well as **Edit** *(reference)* mode.

## How to Use

- add `(foo:-1.0)` in the `positive prompt` to **remove** a concept
- add `(bar:-1.0)` in the `negative prompt` to **enforce** a concept

> [!NOTE]
> **Krea 2** normally drops every prompt weight as soon as a reference image is attached. While this Extension is active, the weights of an **Edit** prompt are honored again, meaning that regular emphasis *(eg. `(foo:1.5)`)* takes effect as well.

> [!TIP]
> On **Krea 2** the weight is a dial, not a switch. A single-stream transformer also attends over the reference and image streams, so one text token carries far less of the attention than it does in **SD**, and `-1.0` can be too subtle to see. Turn it up *(eg. `(foo:-3.0)`)* until the concept goes.

> [!TIP]
> **Krea 2** does not work with the `None` **Emphasis Mode**, which reads a weight as literal characters rather than parsing it. The console says so when that happens; every other mode is fine.

## Console

The Extension reports what it did, so that nothing it decided is silent:

| | |
| --- | --- |
| `NegPiP Loaded` | it is installed and being called, listing the models it can handle |
| `NegPiP Active` | it engaged, naming the model it recognised |
| `NegPiP Enable` | how many tokens it is negating |
| `NegPiP Applied` | *(Krea 2)* the negation reached the transformer |
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
