# 🧩 TiledWan ComfyUI Node Set

Wan2.1 Vace can perform video inpainting on 832x432 81-frame videos. This custom node set and node adapts it to process as long and as large videos as one wants with tiling while maintaining consistency.
One can find a workflow example in the folder "workflow". The workflow is rather comprehensively commented and contains important tips and tricks.

Provide a video, a mask, a prompt and a ref image. The workflow computes itself how it should tile and process the video and it stitches everything back before saving the final output.

Workflow output is meant to be recomposited. It can provide a good workbase for VFX-artists but is far from satisfying out of the box for most standards

## 🎬 Example

https://private-user-images.githubusercontent.com/127590018/477613519-d7e0b9a2-23af-497b-b345-6a6a4217a1aa.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTUwOTc3MzAsIm5iZiI6MTc1NTA5NzQzMCwicGF0aCI6Ii8xMjc1OTAwMTgvNDc3NjEzNTE5LWQ3ZTBiOWEyLTIzYWYtNDk3Yi1iMzQ1LTZhNmE0MjE3YTFhYS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwODEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDgxM1QxNTAzNTBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jN2NiNzBiN2ExYTk1ZmU0MzI4NTgyMTkzMmY4ZjA0MzUzMWQ1YjQ2NzEyZDE5YmQ1NzhhZGJjNWM1N2UzMDgyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9._1K_SL3mjii4WQICOpmOYXG2QNGD4Cmqo6RA12PvbrY


*Process example on one spatial tile and 4 temporal tiles stitched together.*

I provided the inputs files in /Medias for testing. 
Example taken from [Galoi's Will](https://youtu.be/_DAqWS7MyEw).

## ⚙️ Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
git clone https://github/Baverne/comfyUI-TiledWan
```

2. Restart ComfyUI

OR

Install it from the [Manager](https://github.com/Comfy-Org/ComfyUI-Manager)

## 🛠️ Custom Nodes

### 🧩 TiledWan Video VACE pipe

TiledWan Video VACE pipe is designed to run Wan2.1 VACE using tiled processing. It handles large and long videos by splitting them into manageable tiles, processing each tile, and then seamlessly stitching them back together.
Temporal and spatial consistencies are ensured by overwriting overlapping areas and reference inheritance.

### ✂️ Inpaint Crop & Stitch

Two modified [lquesada's](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) nodes that crop around the provided mask area to help the model focus on what matters.
They have been modified to ignore size variation (leads to inconsistencies) and to handle mask apparition and disappearance.

### 🖼️ Image to Mask

Modified [comfy_extras](https://github.com/comfyanonymous/ComfyUI) node which converts image into mask and allows to perform mask normalization and clamping.



## 🚧 Limitations

Even if one can achieve rather good consistency with this workflow, the Wan2.1 vace model does suffer from poor definition and "cartoonish" outputs sometimes.
Increasing spatial tiles number can help but might lead to model cluelessness over proper context. Indeed Wan2.1 is meant to be provided meaningful frames.


## 🙏 Acknowledgements

The workflow was inspired by [Mickmumpitz's](https://www.patreon.com/posts/shoot-entire-ai-127894905?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator&utm_content=join_link).

The node set includes two modified [lquesada's](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) nodes and one from [comfy_extras](https://github.com/comfyanonymous/ComfyUI), all of them licensed under GNU GENERAL PUBLIC LICENSE Version 3. 

# 📄 License
GNU GENERAL PUBLIC LICENSE Version 3, see [LICENSE](LICENSE)