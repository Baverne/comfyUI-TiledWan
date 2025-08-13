# TiledWan ComfyUI Node Set

Some key custom nodes and node adaptations to perform efficient tiled Wan2.1 Inpainting in Comfy-UI.
One can find a workflow example in the folder "worklow". The workflow is rather comprehensively commented and contains important tips and tricks. I suggest to read it all before using.

The workflow has been though to provide an output to be recompositited. It can provide a good workbase for VFX-artists but are far from satisfaying out of the box for most standards.

<table>
    <tr>
        <th>Before</th>
        <th>After</th>
    </tr>
    <tr>
        <td>
            <video src="Medias/BEFORE.mp4" controls width="320"></video>
        </td>
        <td>
            <video src="Medias/AFTER.mp4" controls width="320"></video>
        </td>
    </tr>
</table>

<video src="Medias/BEFORE.mp4" controls width="320"></video>

## Custom Nodes

### TiledWan Video VACE pipe

TiledWan Video VACE pipe is designed to run Wan2.1 VACE using tiled processing. It handles large and long videos by splitting them into manageable tiles, processing each tile, and then seamlessly stitching them back together.
Temporal and spatial consistencies are ensure by a overwrtting overlaping areas and reference inheritance.

### Inpaint Crop & Stich

Two modified [lquesada's](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) nodes that crop arround the provided mask area to help the model focus on what matters.
They have been modified to ignore size variation (leads to inconsistencies) and to handle mask apparition and disparitions.

### Image to Mask

Modified [comfy_extras](https://github.com/comfyanonymous/ComfyUI) node which convert image into mask and allow to perfom mask normalisation and clamping.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github/Baverne/comfyUI-TiledWan comfyUI-TiledWan
```

2. Restart ComfyUI

## Limitations

Even if one can achieve rather good consistency with this workflow, the Wan2.1 vace model do suffer from poor definiton and "cartoonish" outputs sometimes.
Increasing spatial tiles number can help sometimes but might lead to model cluelessness over proper context. Indeed Wan2.1 is meant to be provided meaningful frames.


## Aknowledgemeents

The workflow was inspired by [Mickmumpitz's](https://www.patreon.com/posts/shoot-entire-ai-127894905?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator&utm_content=join_link).

The node set inlcudes two modified [lquesada's](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) nodes and one from [comfy_extras](https://github.com/comfyanonymous/ComfyUI) all of them licensed under GNU GENERAL PUBLIC LICENSE Version 3. 

# License
GNU GENERAL PUBLIC LICENSE Version 3, see [LICENSE](LICENSE)