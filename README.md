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

### TiledWan Image Statistics
Analyze images and display comprehensive statistics including min, max, mean, variance, median and distribution analysis.

**Inputs:**
- `image`: IMAGE - Input image to analyze
- `show_per_channel`: BOOLEAN - Whether to show per-channel statistics for RGB/RGBA images (default: True)

**Outputs:**
- `image`: IMAGE - The original image (pass-through)
- `min_value`: FLOAT - Minimum pixel value in the image
- `max_value`: FLOAT - Maximum pixel value in the image
- `mean_value`: FLOAT - Average pixel value
- `variance`: FLOAT - Variance of pixel values
- `median_value`: FLOAT - Median pixel value

**Features:**
- Comprehensive statistics: min, max, mean, median, variance, standard deviation
- Per-channel analysis for multi-channel images
- Value distribution histogram (10 bins)
- Percentile analysis (1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th)
- Memory usage and image format information
- Automatic issue detection (negative values, out-of-range values, constant images)
- Console output with detailed formatting for easy reading

### TiledWan Mask Statistics
Analyze masks and display comprehensive statistics including coverage, density, connected components and distribution analysis.

**Inputs:**
- `mask`: MASK - Input mask to analyze
- `analyze_connected_components`: BOOLEAN - Whether to perform connected components analysis (default: True)

**Outputs:**
- `mask`: MASK - The original mask (pass-through)
- `min_value`: FLOAT - Minimum pixel value in the mask
- `max_value`: FLOAT - Maximum pixel value in the mask
- `mean_value`: FLOAT - Average pixel value
- `variance`: FLOAT - Variance of pixel values
- `median_value`: FLOAT - Median pixel value
- `white_pixel_count`: INT - Number of pixels considered "white" (>0.5)

**Features:**
- Standard statistics: min, max, mean, median, variance, standard deviation
- Mask-specific analysis: white/black pixel counts, coverage percentage, density classification
- Value distribution: exact counts for few unique values, histogram for complex masks
- Connected components analysis: estimated number and size of separate mask regions
- Mask quality assessment: binary mask detection, uniformity analysis
- Automatic issue detection: negative values, out-of-range values, sparse/dense masks

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

### TiledWan Image Statistics
The TiledWan Image Statistics node can be found in the "TiledWan" category in the ComfyUI node browser.

1. Connect any image to the input
2. Enable or disable per-channel analysis based on your needs
3. Run the workflow - statistics will be displayed in the ComfyUI console
4. The node also outputs individual statistic values that can be used by other nodes

**Console Output includes:**
- **Basic Info**: Image dimensions, total pixels, memory usage, data type
- **Global Statistics**: Min, max, mean, median, variance, standard deviation
- **Per-Channel Analysis**: Individual statistics for R, G, B, A channels
- **Distribution Analysis**: 10-bin histogram showing value distribution
- **Percentiles**: Key percentile values for detailed distribution analysis
- **Issue Detection**: Automatic warnings for potential problems

**Use Cases:**
- **Quality Control**: Detect images with unusual value ranges or distributions
- **Debugging**: Understand what's happening to your images in complex workflows
- **Optimization**: Identify images that might need preprocessing or normalization
- **Analysis**: Extract numerical statistics for further processing or decision making

**Tips:**
- Enable per-channel analysis for color images to understand channel balance
- Use the output values in mathematical nodes for dynamic workflow control
- Check the console output for warnings about potential image issues
- The pass-through design allows you to insert this node anywhere without breaking your workflow

### TiledWan Mask Statistics
The TiledWan Mask Statistics node can be found in the "TiledWan" category in the ComfyUI node browser.

1. Connect any mask to the input
2. Enable or disable connected components analysis based on your needs
3. Run the workflow - statistics will be displayed in the ComfyUI console
4. The node also outputs individual statistic values and white pixel count for further processing

**Console Output includes:**
- **Basic Info**: Mask dimensions, total pixels, memory usage, data type
- **Global Statistics**: Min, max, mean, median, variance, standard deviation
- **Mask-Specific Analysis**: White/black pixel counts, coverage percentage, density classification
- **Value Distribution**: Exact counts for simple masks or histogram for complex ones
- **Connected Components**: Estimated number and size of separate mask regions
- **Quality Assessment**: Binary mask detection, uniformity warnings

**Use Cases:**
- **Mask Quality Control**: Verify mask coverage and density
- **Segmentation Analysis**: Understand mask complexity and component distribution  
- **Workflow Optimization**: Use coverage data to skip processing on empty masks
- **Debugging**: Identify issues with mask generation or processing
- **Data Analysis**: Extract numerical mask properties for decision making

**Tips:**
- Enable connected components analysis for segmentation masks to understand object count
- Use the white_pixel_count output to conditionally process based on mask coverage
- Perfect binary masks (0 and 1 only) will be automatically detected and reported
- The pass-through design allows insertion anywhere in mask processing workflows
- Check console warnings for potential mask quality issues