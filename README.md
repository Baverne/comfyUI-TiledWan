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

### TiledWan Image To Mask
Converts an image to a mask by extracting a specific color channel (red, green, blue, or alpha) with optional clamping and normalization.

**Inputs:**
- `image`: IMAGE - Input image to convert to mask
- `channel`: COMBO - Channel to extract ("red", "green", "blue", "alpha")
- `clamp_output`: BOOLEAN - Whether to clamp output values between 0 and 1 (default: True)
- `normalize_output`: BOOLEAN - Whether to normalize output to 0-1 range (default: True)

**Outputs:**
- `mask`: MASK - The extracted channel as a mask

**Features:**
- Supports RGBA images (if alpha channel is requested but doesn't exist, returns a fully opaque mask)
- Optional normalization: scales the mask values to use the full 0-1 range
- Optional clamping: ensures all values are between 0 and 1
- Robust handling of different image formats

### TiledWan Image Blend
Blend two images using various blend modes with advanced clamping options for negative values.

**Inputs:**
- `image1`: IMAGE - Base/background image
- `image2`: IMAGE - Overlay/foreground image (automatically resized to match image1 if needed)
- `blend_factor`: FLOAT - Blending strength from 0.0 to 1.0 (default: 0.5)
- `blend_mode`: COMBO - Blending mode: "normal", "multiply", "screen", "overlay", "soft_light", "difference"
- `clamp_negative`: BOOLEAN - Whether to clamp values below 0 to 0 (default: False)

**Outputs:**
- `image`: IMAGE - The blended result image

**Features:**
- Automatic image resizing: image2 is automatically scaled to match image1's dimensions
- Multiple blend modes with proper mathematical implementations
- Advanced clamping control: choose whether to allow negative values or clamp them to 0
- Device-aware processing: handles GPU/CPU tensor placement automatically
- Alpha channel handling: properly manages transparency in input images

## Usage

### TiledWan Image To Mask
The TiledWan Image To Mask node can be found in the "TiledWan" category in the ComfyUI node browser. 

1. Connect an image to the input
2. Select which channel to extract (red, green, blue, or alpha)
3. Configure clamping and normalization options as needed
4. The node will output a mask based on the selected channel

**Tips:**
- Use normalization when you want to maximize contrast in the resulting mask
- Use clamping to ensure compatibility with other ComfyUI nodes that expect 0-1 values
- The alpha channel option works even with RGB images (will create a fully opaque mask)

### TiledWan Image Blend
The TiledWan Image Blend node can be found in the "TiledWan" category in the ComfyUI node browser.

1. Connect a base image to `image1` input
2. Connect an overlay image to `image2` input (will be automatically resized if needed)
3. Set the `blend_factor` to control the strength of the blend (0.0 = only image1, 1.0 = full blend effect)
4. Choose a `blend_mode` for different visual effects
5. Enable `clamp_negative` if you want to prevent negative values (useful for certain blend modes)

**Blend Modes:**
- **Normal**: Simple overlay replacement
- **Multiply**: Darkens by multiplying colors
- **Screen**: Lightens by inverting, multiplying, then inverting again
- **Overlay**: Combines multiply and screen based on base color
- **Soft Light**: Subtle lighting effect
- **Difference**: Shows absolute difference between images

**Tips:**
- Use `clamp_negative: False` when working with difference mode to preserve negative values for further processing
- Enable `clamp_negative: True` for final output to ensure valid image data
- Experiment with different blend modes for creative effects