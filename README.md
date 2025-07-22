# TiledWan ComfyUI Node Set

A custom node set for ComfyUI that provides tiled processing capabilities.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone <your-repo-url> comfyUI-TiledWan
```

2. Restart ComfyUI

## Nodes

### TiledWan Pass Through
A simple pass-through node that outputs the same image it receives as input. This serves as a dummy/template node for the TiledWan node set.

**Inputs:**
- `image`: IMAGE - Input image to pass through

**Outputs:**
- `image`: IMAGE - The same image as received in input

## Usage

The TiledWan Pass Through node can be found in the "TiledWan" category in the ComfyUI node browser. Simply connect an image to its input and it will output the same image unchanged.