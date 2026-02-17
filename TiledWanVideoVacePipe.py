import torch
import comfy.utils
import comfy.model_management
import node_helpers
import random
import os
import sys
import importlib
import math
import nodes
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, binary_closing, binary_fill_holes


class Tile:
    """
    Represents a single tile with its content and position in the tiling grid
    """
    def __init__(self, content, temporal_index, line_index, column_index, 
                 temporal_range, spatial_range_h, spatial_range_w):
        self.content = content  # The actual tile tensor
        self.temporal_index = temporal_index  # Index in temporal dimension
        self.line_index = line_index  # Index in height/line dimension
        self.column_index = column_index  # Index in width/column dimension
        
        # Store the ranges for overlap calculations
        self.temporal_range = temporal_range  # (start_frame, end_frame)
        self.spatial_range_h = spatial_range_h  # (start_h, end_h)
        self.spatial_range_w = spatial_range_w  # (start_w, end_w)
    
    def __str__(self):
        return f"Tile[T{self.temporal_index}:L{self.line_index}:C{self.column_index}] " \
               f"Shape: {self.content.shape} " \
               f"Temporal: {self.temporal_range} " \
               f"Spatial: H{self.spatial_range_h} W{self.spatial_range_w}"

class TiledWanVideoVACEpipe:
    """
    Dimension-wise Tiled WanVideo VACE Pipeline - All in one node for processing large videos.
    
    This node combines a dimension-wise tiling system from TileAndStitchBack with 
    the complete WanVideo VACE pipeline. It handles large videos using the advanced algorithm:
    
    1. Temporal tiling: Split video into chunks with overlap (default: 81 frames, 10-frame overlap)
    2. Spatial tiling: Split each chunk into tiles with overlap (default: 832×480, 20-pixel overlap)
    3. WanVideo VACE processing: Process each tile through the complete pipeline
    4. Dimension-wise stitching:
       a) Column-wise: Stitch tiles vertically within each temporal chunk
       b) Line-wise: Stitch columns horizontally to complete temporal chunks  
       c) Temporal: Stitch temporal chunks together across time
    5. Cropping: Final video cropped to exact input dimensions
    6. Memory management: Proper model offloading between tiles to prevent leaks
    
    This enables processing of arbitrarily large videos that would otherwise exceed memory limits,
    while maintaining WanVideo's optimal performance and eliminating overlap artifacts through
    sophisticated dimension-wise fade blending.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Complete inputs combining tiling parameters with WanVideo VACE pipeline
        """
        return {
            "required": {
                # Input video and mask
                "video": ("IMAGE", {"tooltip": "Input video frames to process. Can be very large videos that will be automatically tiled for memory efficiency."}),
                "mask": ("MASK", {"tooltip": "Mask defining areas to be processed. Should match the video dimensions and frame count."}),
                
                # Core WanVideo pipeline inputs
                "model": ("WANVIDEOMODEL", {"tooltip": "WanVideo model for video generation/processing."}),
                "vae": ("WANVAE", {"tooltip": "Video VAE for encoding/decoding frames."}),
                
                # Tiling parameters
                "target_frames": ("INT", {"default": 81, "min": 16, "max": 200, "step": 1, "tooltip": "Target number of frames per temporal chunk. The model works best with 81 frames. If different, might generate artefact or frame drops"}),
                "target_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Target width for spatial tiles. Works best around 832."}),
                "target_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8, "tooltip": "Target height for spatial tiles. Works best around 480."}),
                "frame_overlap": ("INT", {"default": 10, "min": 0, "max": 40, "step": 1, "tooltip": "Number of overlapping frames between temporal chunks."}),
                "spatial_overlap": ("INT", {"default": 20, "min": 0, "max": 100, "step": 4, "tooltip": "Pixel overlap between spatial tiles."}),
                
                # WanVideoSampler core parameters
                "steps": ("INT", {"default": 30, "min": 1, "tooltip": "Number of diffusion sampling steps."}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "Classifier-free guidance strength. Higher values follow the conditioning more closely."}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "Shift parameter for the sampling schedule."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducible results. Use the same seed to get identical outputs."}),
                "scheduler": (["unipc", "unipc/beta", "dpm++", "dpm++/beta","dpm++_sde", "dpm++_sde/beta", "euler", "euler/beta", "euler/accvideo", "deis", "lcm", "lcm/beta", "flowmatch_causvid", "flowmatch_distill", "multitalk"],
                    {"default": 'unipc', "tooltip": "Sampling scheduler algorithm."}),
                
                # WanVideoVACEEncode parameters  
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "VACE encoding strength. Controls how much the video is modified during processing. 1.0 mean total reconstruction, 0.0 means no change."}),
                "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # WanVideoDecode parameters
                "decode_enable_vae_tiling": ("BOOLEAN", {"default": False}),
                "decode_tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8}),
                "decode_tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8}),
                
                # Processing parameters
                "debug_mode": ("BOOLEAN", {"default": True}),
                "debug_color_shift": ("BOOLEAN", {"default": False, "tooltip": "Enable color shift debugging to visualize tile boundaries. Useful for debugging tiling artifacts."}),
                "force_offload_between_tiles": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # WanVideoSampler optional inputs
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "samples": ("LATENT",),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "batched_cfg": ("BOOLEAN", {"default": False}),
                "rope_function": (["default", "comfy"], {"default": "comfy"}),
                
                # WanVideoSampler advanced optional inputs
                "feta_args": ("FETAARGS",),
                "context_options": ("CONTEXTOPTIONS",),
                "loop_args": ("LOOPARGS",),
                "sigmas": ("SIGMAS",),
                "unianimate_poses": ("UNIANIMATE_POSES",),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS",),
                "uni3c_embeds": ("UNI3C_EMBEDS",),
                "multitalk_embeds": ("MULTITALK_EMBEDS",),
                "freeinit_args": ("FREEINIT_ARGS",),
                
                # External pre-built arguments (for modularity)
                "cache_args": ("CACHEARGS",),
                "slg_args": ("SLGARGS",),
                "experimental_args": ("EXPERIMENTALARGS",),
                
                # WanVideoVACEEncode optional inputs  
                "vace_ref_images": ("IMAGE",),
                "vace_tiled_vae": ("BOOLEAN", {"default": False}),
                
                # WanVideoDecode optional inputs
                "decode_normalization": (["default", "minmax"], {"default": "default"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_video", "processing_info")
    OUTPUT_TOOLTIPS = ("Processed video with WanVideo VACE pipeline applied through tiled processing. Maintains original dimensions and quality.", 
                      "Detailed information about the processing including tile counts, timing, and processing summary.")
    FUNCTION = "process_tiled_wanvideo"
    OUTPUT_NODE = True
    CATEGORY = "TiledWan"
    DESCRIPTION = """
    Wan2.1 VACE video processing node that applies WanVideo VACE pipeline to large videos through tiling.

    Enables processing of arbitrarily large videos that would otherwise exceed memory or models limits.
    Uses dimension-wise tiling with temporal and spatial overlap to maintain quality and consistency.
    The complete WanVideo VACE pipeline is applied to each tile with overwriting on overlapping regions for consistency.

    Processing Algorithm:
    1. Temporal tiling: Split video into chunks with overlap (default: 81 frames, 10-frame overlap)
    2. Spatial tiling: Split each chunk into tiles with overlap (default: 832×480, 20-pixel overlap)  
    3. Temporal consistency: Previous chunks provide reference frames for upcoming chunks
    4. Spatial consistency: Already-processed neighboring tiles overwrite overlapping regions
    5. WanVideo VACE processing: Each tile processed through complete pipeline
    6. Dimension-wise stitching: Column-wise → Line-wise → Temporal stitching
    7. Final cropping: Output matches exact input dimensions

    Key Features:
    - Handles any video size through tiling
    - Temporal consistency across chunks via frame reference chaining
    - Spatial consistency through neighbor tile overwriting
    - Memory-efficient with model offloading between tiles
    - Complete WanVideo VACE pipeline integration
    - Fade blending for seamless stitching
    - Tensor safety with hard copies to prevent data contamination
    
    Consistency Mechanisms:
    - Temporal: Last frames from previous chunks overwrite first frames of current chunks
    - Spatial: Left/top neighbors overwrite overlapping edges in current tiles
    - All overwritten regions have masks zeroed for proper processing
    - Reference frames are used to maintain temporal coherence across temporal chunks
    """

    def process_tiled_wanvideo(self, video, mask, model, vae, target_frames, target_width, target_height, 
                              frame_overlap, spatial_overlap, steps, cfg, shift, seed, scheduler,
                              vace_strength, vace_start_percent, vace_end_percent, 
                              decode_enable_vae_tiling, decode_tile_x, decode_tile_y,
                              decode_tile_stride_x, decode_tile_stride_y, debug_mode, debug_color_shift,
                              force_offload_between_tiles, **kwargs):
        """
        Process large video through tiled WanVideo VACE pipeline with dimension-wise stitching
        """
        
        print("\n" + "="*80)
        print("              TILED WANVIDEO VACE PIPELINE (DIMENSION-WISE)")
        print("="*80)
        print("🚀 Starting dimension-wise tiled WanVideo VACE processing...")
        
        try:
            # Import WanVideo nodes
            custom_nodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI-WanVideoWrapper")
            if custom_nodes_path not in sys.path:
                sys.path.append(custom_nodes_path)
            
            parent_path = os.path.dirname(custom_nodes_path)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
            
            package_name = os.path.basename(custom_nodes_path)
            # WanVideoSampler is in nodes_sampler.py
            wanvideo_sampler_package = importlib.import_module(f"{package_name}.nodes_sampler")
            # WanVideoVACEEncode and WanVideoDecode are in nodes.py
            wanvideo_package = importlib.import_module(f"{package_name}.nodes")
            
            WanVideoSampler = wanvideo_sampler_package.WanVideoSampler
            WanVideoVACEEncode = wanvideo_package.WanVideoVACEEncode
            WanVideoDecode = wanvideo_package.WanVideoDecode
            print("✅ WanVideo nodes imported successfully!")
            
            # Input validation and preprocessing
            batch_size, height, width, channels = video.shape
            mask_batch, mask_height, mask_width = mask.shape
            
            print(f"📹 Input video shape: {video.shape} (B×H×W×C)")
            print(f"🎭 Input mask shape: {mask.shape} (B×H×W)")
            print(f"🎯 Target tile size: {target_frames} frames × {target_width}×{target_height}")
            print(f"🔗 Overlaps: {frame_overlap} frames, {spatial_overlap} pixels")
            
            # Validate mask dimensions
            if mask_batch != batch_size or mask_height != height or mask_width != width:
                print(f"⚠️  Warning: Mask dimensions {mask.shape} don't match video {video.shape}")
                print("   Resizing mask to match video...")
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), size=(height, width), mode='nearest'
                ).squeeze(1)
                print(f"✅ Mask resized to: {mask.shape}")
            
            # STEP 1: Calculate tile dimensions using improved algorithm
            print(f"\n📏 STEP 1: Calculating optimal tile layout...")
            temporal_tiles = self._calculate_temporal_tiles(batch_size, target_frames, frame_overlap)
            spatial_tiles_h = self._calculate_spatial_tiles(height, target_height, spatial_overlap) 
            spatial_tiles_w = self._calculate_spatial_tiles(width, target_width, spatial_overlap)
            
            total_tiles = len(temporal_tiles) * len(spatial_tiles_h) * len(spatial_tiles_w)
            print(f"⏱️  Temporal chunks: {len(temporal_tiles)}")
            print(f"🗺️  Spatial tiles per frame: {len(spatial_tiles_h)}×{len(spatial_tiles_w)}")
            print(f"📦 Total tiles to process: {total_tiles}")
            
            # STEP 2: Extract all tiles and process through WanVideo VACE pipeline
            print(f"\n🧩 STEP 2: Extracting and processing {total_tiles} tiles through WanVideo VACE...")
            all_tiles = self._extract_and_process_wanvideo_tiles(
                video, mask, temporal_tiles, spatial_tiles_h, spatial_tiles_w,
                model, vae, WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                steps, cfg, shift, seed, scheduler, vace_strength, vace_start_percent, vace_end_percent,
                decode_enable_vae_tiling, decode_tile_x, decode_tile_y, decode_tile_stride_x, decode_tile_stride_y,
                debug_color_shift, force_offload_between_tiles, debug_mode, frame_overlap, spatial_overlap, kwargs
            )
            
            # STEP 3: Dimension-wise stitching - Column-wise (vertical stitching)
            print(f"\n🔄 STEP 3: Column-wise stitching (vertical)...")
            column_strips = self._stitch_columns(all_tiles, spatial_overlap, debug_mode)
            
            # STEP 4: Dimension-wise stitching - Line-wise (horizontal stitching)  
            print(f"\n🔄 STEP 4: Line-wise stitching (horizontal)...")
            temporal_chunks = self._stitch_lines(column_strips, spatial_overlap, debug_mode)
            
            # STEP 5: Dimension-wise stitching - Temporal stitching
            print(f"\n🔄 STEP 5: Temporal stitching...")
            stitched_video = self._stitch_temporal_chunks_new(temporal_chunks, temporal_tiles, frame_overlap)
            
            # STEP 6: Crop to original dimensions (ensure exact input size)
            print(f"\n✂️ STEP 6: Cropping to original dimensions...")
            print(f"📐 Stitched video shape: {stitched_video.shape}")
            print(f"🎯 Target shape: {video.shape}")
            final_video = stitched_video[:batch_size, :height, :width, :channels]
            print(f"✂️ Cropped to: {final_video.shape}")
            
            # Generate comprehensive processing info
            processing_summary = self._generate_wanvideo_processing_summary(
                all_tiles, temporal_tiles, spatial_tiles_h, spatial_tiles_w, total_tiles
            )
            
            print(f"✅ Dimension-wise tiled WanVideo VACE processing completed!")
            print(f"📤 Final video shape: {final_video.shape}")
            print(f"🧩 Total tiles processed: {len(all_tiles)}")
            successful_tiles = sum(1 for tile in all_tiles if hasattr(tile, 'processing_status') and tile.processing_status == 'success')
            print(f"✅ Successful tiles: {successful_tiles}/{total_tiles}")
            print("="*80 + "\n")
            
            # CLEANUP: Clear variables to prevent interference with subsequent runs
            try:
                del all_tiles, column_strips, temporal_chunks, stitched_video
                del temporal_tiles, spatial_tiles_h, spatial_tiles_w
                if 'previous_chunk_stitched_frame' in locals():
                    del previous_chunk_stitched_frame
                print("🧹 Variables cleaned up for next run")
            except:
                pass
            
            return (final_video, processing_summary)
            
        except Exception as e:
            print(f"❌ Error in tiled WanVideo VACE pipeline: {str(e)}")
            print(f"📋 Full traceback:")
            import traceback
            print(traceback.format_exc())
            print("="*80 + "\n")
            
            # Return original video in case of error
            error_info = f"Error during tiled WanVideo processing: {str(e)}"
            
            # CLEANUP: Clear variables even on error to prevent interference with subsequent runs
            try:
                if 'all_tiles' in locals():
                    del all_tiles
                if 'column_strips' in locals():
                    del column_strips
                if 'temporal_chunks' in locals():
                    del temporal_chunks
                if 'temporal_tiles' in locals():
                    del temporal_tiles
                if 'spatial_tiles_h' in locals():
                    del spatial_tiles_h
                if 'spatial_tiles_w' in locals():
                    del spatial_tiles_w
                if 'previous_chunk_stitched_frame' in locals():
                    del previous_chunk_stitched_frame
                print("🧹 Variables cleaned up after error")
            except:
                pass
            
            return (video, error_info)
    
    
    def _extract_and_process_wanvideo_tiles(self, video, mask, temporal_tiles, spatial_tiles_h, spatial_tiles_w,
                                        model, vae, WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                                        steps, cfg, shift, seed, scheduler, vace_strength, vace_start_percent, vace_end_percent,
                                        decode_enable_vae_tiling, decode_tile_x, decode_tile_y, decode_tile_stride_x, decode_tile_stride_y,
                                        debug_color_shift, force_offload_between_tiles, debug_mode, frame_overlap, spatial_overlap, kwargs):
        """
        Extract all tiles and process through WanVideo VACE pipeline with frame-wise temporal consistency
        """
        all_tiles = []
        
        # Store previous temporal chunk's COMPLETE STITCHED FRAME for reference
        # IMPORTANT: Always start fresh to avoid interference from previous runs
        previous_chunk_stitched_frame = None
        
        for temporal_idx, (t_start, t_end) in enumerate(temporal_tiles):
            if debug_mode:
                print(f"   🎬 Processing temporal chunk {temporal_idx + 1}/{len(temporal_tiles)} (frames {t_start}-{t_end-1})")
            
            # Extract temporal chunk
            video_chunk = video[t_start:t_end].clone()  # HARD COPY to avoid modifying original
            mask_chunk = mask[t_start:t_end].clone()    # HARD COPY to avoid modifying original
            
            # Store current chunk tiles for stitching the reference frame
            current_chunk_tiles = []
            
            # Process each spatial tile within this temporal chunk
            for h_idx, (h_start, h_end) in enumerate(spatial_tiles_h):
                for w_idx, (w_start, w_end) in enumerate(spatial_tiles_w):
                    tile_idx = len(all_tiles)  # Global tile index
                    
                    if debug_mode:
                        print(f"      🧩 Tile {tile_idx + 1}: T{temporal_idx}:L{h_idx}:C{w_idx} "
                            f"H[{h_start}:{h_end}] × W[{w_start}:{w_end}]")
                    
                    # Extract spatial tiles from temporal chunk
                    video_tile = video_chunk[:, h_start:h_end, w_start:w_end, :].clone()  # HARD COPY to avoid modifying chunk
                    mask_tile = mask_chunk[:, h_start:h_end, w_start:w_end].clone()        # HARD COPY to avoid modifying chunk
                    
                    # ========== TEMPORAL OVERWRITING FOR CONSISTENCY ==========
                    video_tile, mask_tile = self._apply_temporal_overwriting(
                        video_tile, mask_tile, temporal_idx, temporal_tiles, frame_overlap,
                        previous_chunk_stitched_frame, t_start, h_start, h_end, w_start, w_end, debug_mode
                    )

                    # ========== SPATIAL OVERWRITING FOR CONSISTENCY ==========
                    # Apply left neighbor overwriting
                    video_tile, mask_tile = self._apply_spatial_overwriting_left(
                        video_tile, mask_tile, temporal_idx, h_idx, w_idx, w_start, current_chunk_tiles, debug_mode
                    )
                    
                    # Apply top neighbor overwriting
                    video_tile, mask_tile = self._apply_spatial_overwriting_top(
                        video_tile, mask_tile, temporal_idx, h_idx, w_idx, h_start, current_chunk_tiles, debug_mode
                    )
                    
                    # Apply top-left corner overwriting
                    video_tile, mask_tile = self._apply_spatial_overwriting_corner(
                        video_tile, mask_tile, temporal_idx, h_idx, w_idx, h_start, w_start, current_chunk_tiles, debug_mode
                    )
                    
                    # ========== FRAME-WISE TEMPORAL CONSISTENCY ==========
                    tile_ref_images = self._determine_tile_reference_images(
                        temporal_idx, temporal_tiles, frame_overlap, previous_chunk_stitched_frame,
                        h_idx, w_idx, debug_mode, kwargs
                    )
                    
                    # Process tile through WanVideo VACE pipeline with frame-wise temporal reference
                    try:
                        # Create kwargs with tile-specific reference image (COMPLETE FRAME)
                        tile_kwargs = kwargs.copy()
                        tile_kwargs["vace_ref_images"] = tile_ref_images
                        
                        processed_tile, tile_latents = self._process_tile_through_wanvideo(
                            video_tile, mask_tile, model, vae, 
                            WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                            steps, cfg, shift, seed + tile_idx,  # Unique seed per tile
                            scheduler, vace_strength, vace_start_percent, vace_end_percent,
                            decode_enable_vae_tiling, decode_tile_x, decode_tile_y,
                            decode_tile_stride_x, decode_tile_stride_y, debug_color_shift, tile_kwargs
                        )
                        
                        # Create Tile object with processed content
                        tile = Tile(
                            content=processed_tile,
                            temporal_index=temporal_idx,
                            line_index=h_idx,
                            column_index=w_idx,
                            temporal_range=(t_start, t_end),
                            spatial_range_h=(h_start, h_end),
                            spatial_range_w=(w_start, w_end)
                        )
                        
                        # Store processing status and reference info for tracking
                        tile.processing_status = 'success'
                        tile.seed_used = seed + tile_idx
                        tile.latents = tile_latents
                        tile.ref_frame_source = f"complete_frame_T{temporal_idx-1}" if temporal_idx > 0 and previous_chunk_stitched_frame is not None else "user_provided"
                        
                        all_tiles.append(tile)
                        current_chunk_tiles.append(tile)
                        
                        # Force memory cleanup between tiles if requested
                        if force_offload_between_tiles:
                            self._force_memory_cleanup(model, vae)
                        
                        # CLEANUP: Delete tile copies to free memory immediately
                        try:
                            del video_tile, mask_tile, processed_tile
                            if 'tile_latents' in locals():
                                del tile_latents
                        except:
                            pass
                            
                    except Exception as tile_error:
                        print(f"         ❌ Error processing tile {tile_idx + 1}: {str(tile_error)}")
                        
                        # Create fallback Tile object with original content
                        tile = Tile(
                            content=video_tile,  # Use original video tile as fallback
                            temporal_index=temporal_idx,
                            line_index=h_idx,
                            column_index=w_idx,
                            temporal_range=(t_start, t_end),
                            spatial_range_h=(h_start, h_end),
                            spatial_range_w=(w_start, w_end)
                        )
                        
                        # Store error status for tracking
                        tile.processing_status = f'failed: {str(tile_error)}'
                        tile.seed_used = seed + tile_idx
                        tile.latents = None
                        tile.ref_frame_source = "error_fallback"
                        
                        all_tiles.append(tile)
                        current_chunk_tiles.append(tile)
                        
                        # CLEANUP: Delete tile copies to free memory immediately (error case)
                        try:
                            del video_tile, mask_tile
                        except:
                            pass
            
            # STITCH THE CURRENT CHUNK TO CREATE REFERENCE FOR NEXT CHUNK
            if debug_mode:
                print(f"      🔧 Stitching current chunk T{temporal_idx} for next chunk's reference...")
            
            # Group current chunk tiles for stitching
            current_chunk_tiles_for_stitching = [tile for tile in current_chunk_tiles]
            
            # Perform dimension-wise stitching for this temporal chunk
            try:
                # Step 1: Column-wise stitching
                column_strips = self._stitch_columns(current_chunk_tiles_for_stitching, spatial_overlap, False)  # No debug for reference stitching
                
                # Step 2: Line-wise stitching  
                temporal_chunks = self._stitch_lines(column_strips, spatial_overlap, False)  # No debug for reference stitching
                
                # Get the stitched chunk content (should be only one chunk)
                if temporal_chunks and len(temporal_chunks) > 0:
                    current_chunk_stitched = temporal_chunks[0]['content']
                    
                    # Store this stitched chunk as reference for next temporal chunk
                    previous_chunk_stitched_frame = current_chunk_stitched
                    
                    if debug_mode:
                        print(f"      ✅ Chunk T{temporal_idx} stitched for reference: {current_chunk_stitched.shape}")
                else:
                    print(f"      ⚠️  Warning: Failed to stitch chunk T{temporal_idx} for reference")
                    previous_chunk_stitched_frame = None
                    
            except Exception as stitching_error:
                print(f"      ❌ Error stitching chunk T{temporal_idx} for reference: {str(stitching_error)}")
                previous_chunk_stitched_frame = None
            
            if debug_mode:
                chain_status = "user_ref" if temporal_idx == 0 else f"complete_frame_from_T{temporal_idx-1}"
                print(f"      ✅ Temporal chunk {temporal_idx} completed with {chain_status} references")
            
            # CLEANUP: Delete temporal chunk copies to free memory
            try:
                del video_chunk, mask_chunk
                del current_chunk_tiles, current_chunk_tiles_for_stitching
                if 'column_strips' in locals():
                    del column_strips
                if 'temporal_chunks' in locals():
                    del temporal_chunks
                if 'current_chunk_stitched' in locals():
                    del current_chunk_stitched
            except:
                pass
        
        print(f"   ✅ Extracted and processed {len(all_tiles)} tiles through WanVideo VACE pipeline")
        successful_tiles = sum(1 for tile in all_tiles if tile.processing_status == 'success')
        print(f"   🎯 Success rate: {successful_tiles}/{len(all_tiles)} ({(successful_tiles/len(all_tiles))*100:.1f}%)")
        
        # Debug: Print overwriting statistics
        if debug_mode:
            temporal_overwrites = sum(1 for t_idx in range(1, len(temporal_tiles)))
            spatial_overwrites = 0
            corner_overwrites = 0
            
            for temporal_idx in range(len(temporal_tiles)):
                for h_idx in range(len(spatial_tiles_h)):
                    for w_idx in range(len(spatial_tiles_w)):
                        if w_idx > 0:  # Has left neighbor
                            spatial_overwrites += 1
                        if h_idx > 0:  # Has top neighbor
                            spatial_overwrites += 1
                        if h_idx > 0 and w_idx > 0:  # Has corner conflict
                            corner_overwrites += 1
            
            print(f"   🔄 Overwriting consistency enhancements:")
            print(f"      • Temporal overwrites: {temporal_overwrites * len(spatial_tiles_h) * len(spatial_tiles_w)} tile operations")
            print(f"      • Spatial overwrites: {spatial_overwrites} edge operations") 
            print(f"      • Corner overwrites: {corner_overwrites} corner conflict resolutions")
        
        # Debug: Print frame-wise temporal consistency chain summary
        if debug_mode:
            print(f"   🖼️  Frame-wise temporal consistency chain summary:")
            print(f"      • Chunk 0: Used user-provided reference images")
            for t_idx in range(1, len(temporal_tiles)):
                print(f"      • Chunk {t_idx}: Used complete stitched frame from Chunk {t_idx-1}")
        
        # CLEANUP: Ensure previous_chunk_stitched_frame doesn't persist for next run
        try:
            del previous_chunk_stitched_frame
        except:
            pass
        
        return all_tiles
    
    def _process_tile_through_wanvideo(self, video_tile, mask_tile, model, vae,
                                    WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                                    steps, cfg, shift, seed, scheduler, vace_strength,
                                    vace_start_percent, vace_end_percent, decode_enable_vae_tiling,
                                    decode_tile_x, decode_tile_y, decode_tile_stride_x, 
                                    decode_tile_stride_y, debug_color_shift, kwargs):
        """Process a single tile through the complete WanVideo VACE pipeline with tile-specific reference"""
        
        tile_frames, tile_height, tile_width = video_tile.shape[:3]
        
        # Get tile-specific reference images (from temporal consistency chain)
        tile_ref_images = kwargs.get("vace_ref_images")
        
        # Step 1: Create VACE embeds for this tile with its specific reference
        vace_node = WanVideoVACEEncode()
        vace_embeds = vace_node.process(
            vae=vae,
            width=tile_width,
            height=tile_height, 
            num_frames=tile_frames,
            strength=vace_strength,
            vace_start_percent=vace_start_percent,
            vace_end_percent=vace_end_percent,
            input_frames=video_tile,        # Use tile as input frames
            ref_images=tile_ref_images,     # Use tile-specific reference (from chain or user)
            input_masks=mask_tile,          # Use tile mask
            tiled_vae=kwargs.get("vace_tiled_vae", False)
        )[0]
        
        # Step 2: Run WanVideoSampler on this tile
        sampler_node = WanVideoSampler()
        latent_samples = sampler_node.process(
            model=model,
            image_embeds=vace_embeds,
            steps=steps,
            cfg=cfg,
            shift=shift,
            seed=seed,
            scheduler=scheduler,
            riflex_freq_index=kwargs.get("riflex_freq_index", 0),
            text_embeds=kwargs.get("text_embeds"),
            samples=kwargs.get("samples"),
            denoise_strength=kwargs.get("denoise_strength", 1.0),
            force_offload=kwargs.get("force_offload", True),
            
            # External arguments
            cache_args=kwargs.get("cache_args"),
            slg_args=kwargs.get("slg_args"),
            experimental_args=kwargs.get("experimental_args"),
            
            # Other arguments
            feta_args=kwargs.get("feta_args"),
            context_options=kwargs.get("context_options"),
            flowedit_args=None,
            batched_cfg=kwargs.get("batched_cfg", False),
            rope_function=kwargs.get("rope_function", "comfy"),
            loop_args=kwargs.get("loop_args"),
            sigmas=kwargs.get("sigmas"),
            unianimate_poses=kwargs.get("unianimate_poses"),
            fantasytalking_embeds=kwargs.get("fantasytalking_embeds"),
            uni3c_embeds=kwargs.get("uni3c_embeds"),
            multitalk_embeds=kwargs.get("multitalk_embeds"),
            freeinit_args=kwargs.get("freeinit_args")
        )[0]
        
        # Step 3: Decode latents back to video for this tile
        decode_node = WanVideoDecode()
        processed_tile = decode_node.decode(
            vae=vae,
            samples=latent_samples,
            enable_vae_tiling=decode_enable_vae_tiling,
            tile_x=decode_tile_x,
            tile_y=decode_tile_y,
            tile_stride_x=decode_tile_stride_x,
            tile_stride_y=decode_tile_stride_y,
            normalization=kwargs.get("decode_normalization", "default")
        )[0]
        
        # Step 4: Apply debug color shift to identify tiles visually (if enabled)
        if debug_color_shift:
            debug_color_shifted_tile = self._apply_debug_color_shift(processed_tile, seed)
            return debug_color_shifted_tile, latent_samples
        else:
            return processed_tile, latent_samples
    
    def _apply_debug_color_shift(self, tile, seed):
        """Apply a random color shift to each tile for debugging visualization"""
        import random
        
        # Generate truly random RGB shifts (different each time, not based on seed)
        r_shift = (random.random() - 0.5) * 0.3  # ±0.15 range
        g_shift = (random.random() - 0.5) * 0.3  # ±0.15 range  
        b_shift = (random.random() - 0.5) * 0.3  # ±0.15 range
        
        print(f"      🎨 Debug color shift - Tile: R{r_shift:+.3f}, G{g_shift:+.3f}, B{b_shift:+.3f}")
        
        # Apply color shift to the tile
        shifted_tile = tile.clone()
        
        if tile.shape[-1] >= 3:  # Ensure we have RGB channels
            shifted_tile[:, :, :, 0] = torch.clamp(shifted_tile[:, :, :, 0] + r_shift, 0, 1)
            shifted_tile[:, :, :, 1] = torch.clamp(shifted_tile[:, :, :, 1] + g_shift, 0, 1)
            shifted_tile[:, :, :, 2] = torch.clamp(shifted_tile[:, :, :, 2] + b_shift, 0, 1)
        
        return shifted_tile

    def _stitch_columns(self, all_tiles, spatial_overlap, debug_mode):
        """
        Stitch tiles vertically (same column, same temporal chunk) to create column strips
        """
        # Group tiles by temporal_index and column_index
        column_groups = {}
        for tile in all_tiles:
            key = (tile.temporal_index, tile.column_index)
            if key not in column_groups:
                column_groups[key] = []
            column_groups[key].append(tile)
        
        # Sort each group by line_index
        for key in column_groups:
            column_groups[key].sort(key=lambda t: t.line_index)
        
        column_strips = []
        
        for (temporal_idx, col_idx), column_tiles in column_groups.items():
            if debug_mode and temporal_idx == 0:
                print(f"      🔄 Column {col_idx}: {len(column_tiles)} tiles")
            
            # Stitch tiles in this column vertically
            if len(column_tiles) == 1:
                column_strip = column_tiles[0].content
            else:
                column_strip = self._stitch_tiles_vertically(column_tiles, spatial_overlap)
            
            # Store column strip with its position info
            strip_info = {
                'content': column_strip,
                'temporal_index': temporal_idx,
                'column_index': col_idx,
                'spatial_range_w': column_tiles[0].spatial_range_w,
                'spatial_range_h': (column_tiles[0].spatial_range_h[0], column_tiles[-1].spatial_range_h[1])
            }
            column_strips.append(strip_info)
        
        print(f"   ✅ Created {len(column_strips)} column strips")
        return column_strips
    

    def _stitch_lines(self, column_strips, spatial_overlap, debug_mode):
        """
        Stitch column strips horizontally (same temporal chunk) to create complete temporal chunks
        """
        # Group column strips by temporal_index
        temporal_groups = {}
        for strip in column_strips:
            temporal_idx = strip['temporal_index']
            if temporal_idx not in temporal_groups:
                temporal_groups[temporal_idx] = []
            temporal_groups[temporal_idx].append(strip)
        
        # Sort each group by column_index
        for temporal_idx in temporal_groups:
            temporal_groups[temporal_idx].sort(key=lambda s: s['column_index'])
        
        temporal_chunks = []
        
        for temporal_idx, strips in temporal_groups.items():
            if debug_mode and temporal_idx == 0:
                print(f"      🔄 Temporal chunk {temporal_idx}: {len(strips)} strips")
            
            # Stitch column strips horizontally
            if len(strips) == 1:
                chunk_content = strips[0]['content']
            else:
                chunk_content = self._stitch_strips_horizontally(strips, spatial_overlap)
            
            # Store temporal chunk with its info
            chunk_info = {
                'content': chunk_content,
                'temporal_index': temporal_idx,
                'spatial_range_h': strips[0]['spatial_range_h'],
                'spatial_range_w': (strips[0]['spatial_range_w'][0], strips[-1]['spatial_range_w'][1])
            }
            temporal_chunks.append(chunk_info)
        
        # Sort by temporal index
        temporal_chunks.sort(key=lambda c: c['temporal_index'])
        
        print(f"   ✅ Created {len(temporal_chunks)} temporal chunks")
        return temporal_chunks

    def _stitch_tiles_vertically(self, column_tiles, spatial_overlap):
        """
        Stitch tiles vertically (in height dimension) with proper fade blending.
        Always respects spatial_overlap limit
        """
        if len(column_tiles) == 1:
            return column_tiles[0].content
        
        # Get dimensions
        first_tile = column_tiles[0].content
        total_height = sum(tile.spatial_range_h[1] - tile.spatial_range_h[0] for tile in column_tiles)
        # Subtract overlaps
        total_height -= spatial_overlap * (len(column_tiles) - 1)
        
        width = first_tile.shape[2]
        channels = first_tile.shape[3]
        frames = first_tile.shape[0]
        
        # Create result tensor
        result = torch.zeros((frames, total_height, width, channels), 
                           dtype=first_tile.dtype, device=first_tile.device)
        
        current_h = 0
        for i, tile in enumerate(column_tiles):
            tile_h = tile.content.shape[1]
            
            if i == 0:
                # First tile - place entirely
                result[:, current_h:current_h+tile_h, :, :] = tile.content
                current_h += tile_h
            else:
                # Subsequent tiles - use full available overlap for blending
                expected_start = tile.spatial_range_h[0]
                overlap_h  = current_h - expected_start
                
                if overlap_h > 0:
                    # Get regions for blending (using full overlap)
                    result_overlap = result[:, current_h-overlap_h:current_h, :, :]
                    tile_overlap = tile.content[:, :overlap_h, :, :]
                    
                    # Create vertical fade mask
                    fade_mask = self._create_vertical_fade_mask(overlap_h, result_overlap.dtype, result_overlap.device)
                    
                    # Apply fade blending
                    blended_overlap = result_overlap * (1.0 - fade_mask) + tile_overlap * fade_mask
                    result[:, current_h-overlap_h:current_h, :, :] = blended_overlap
                    
                    # Place the non-overlapping part
                    if tile_h > overlap_h:
                        result[:, current_h:current_h+tile_h-overlap_h, :, :] = tile.content[:, overlap_h:, :, :]
                    current_h += tile_h - overlap_h
                else:
                    # No overlap - place directly
                    result[:, current_h:current_h+tile_h, :, :] = tile.content
                    current_h += tile_h
        
        return result
    
    def _stitch_strips_horizontally(self, strips, spatial_overlap):
        """
        Stitch column strips horizontally (in width dimension) with proper fade blending.
        Always respects spatial_overlap limit - starts blending from the correct position
        within the overlapping region when actual overlap exceeds spatial_overlap.
        """
        if len(strips) == 1:
            return strips[0]['content']
        
        # Get dimensions
        first_strip = strips[0]['content']
        total_width = sum(s['spatial_range_w'][1] - s['spatial_range_w'][0] for s in strips)
        # Subtract overlaps
        total_width -= spatial_overlap * (len(strips) - 1)
        
        height = first_strip.shape[1]
        channels = first_strip.shape[3]
        frames = first_strip.shape[0]
        
        # Create result tensor
        result = torch.zeros((frames, height, total_width, channels), 
                           dtype=first_strip.dtype, device=first_strip.device)
        
        current_w = 0
        for i, strip in enumerate(strips):
            strip_content = strip['content']
            strip_w = strip_content.shape[2]
            
            if i == 0:
                # First strip - place entirely
                result[:, :, current_w:current_w+strip_w, :] = strip_content
                current_w += strip_w
            else:
                # Subsequent strips - use full available overlap for blending
                expected_start = strip['spatial_range_w'][0]
                overlap_w = current_w - expected_start

                if overlap_w > 0:
                    # Get regions for blending (using full overlap)
                    result_overlap = result[:, :, current_w-overlap_w:current_w, :]
                    strip_overlap = strip_content[:, :, :overlap_w, :]
                    
                    # Create horizontal fade mask
                    fade_mask = self._create_horizontal_fade_mask(overlap_w, result_overlap.dtype, result_overlap.device)
                    
                    # Apply fade blending
                    blended_overlap = result_overlap * (1.0 - fade_mask) + strip_overlap * fade_mask
                    result[:, :, current_w-overlap_w:current_w, :] = blended_overlap
                    
                    # Place the non-overlapping part
                    if strip_w > overlap_w:
                        result[:, :, current_w:current_w+strip_w-overlap_w, :] = strip_content[:, :, overlap_w:, :]
                    current_w += strip_w - overlap_w
                else:
                    # No overlap - place directly
                    result[:, :, current_w:current_w+strip_w, :] = strip_content
                    current_w += strip_w
        
        return result
    
    def _create_vertical_fade_mask(self, fade_size, dtype, device):
        """Create a vertical fade mask for blending tiles vertically."""
        fade_mask = torch.linspace(0, 1, fade_size, dtype=dtype, device=device)
        # Shape: (1, fade_size, 1, 1) for broadcasting
        return fade_mask.view(1, fade_size, 1, 1)
    
    def _create_horizontal_fade_mask(self, fade_size, dtype, device):
        """Create a horizontal fade mask for blending tiles horizontally."""
        fade_mask = torch.linspace(0, 1, fade_size, dtype=dtype, device=device)
        # Shape: (1, 1, fade_size, 1) for broadcasting
        return fade_mask.view(1, 1, fade_size, 1)
    
    def _stitch_temporal_chunks_new(self, temporal_chunks, temporal_tiles, frame_overlap):
        """
        Stitch temporal chunks to create the final output with temporal blending.
        Always respects frame_overlap limit - starts blending from the correct position
        within the overlapping region when actual overlap exceeds frame_overlap.
        """
        if not temporal_chunks:
            return None
            
        if len(temporal_chunks) == 1:
            return temporal_chunks[0]['content']
        
        # Get dimensions from first chunk
        first_chunk_content = temporal_chunks[0]['content']
        
        # Calculate total frames accounting for overlaps
        total_frames = sum(chunk['content'].shape[0] for chunk in temporal_chunks)
        if frame_overlap > 0:
            total_frames -= (len(temporal_chunks) - 1) * frame_overlap
        
        # Get spatial dimensions from the stitched chunks (not original input)
        height = first_chunk_content.shape[1]
        width = first_chunk_content.shape[2]
        channels = first_chunk_content.shape[3]
        
        print(f"   📏 Temporal stitching dimensions:")
        print(f"      • Total frames: {total_frames}")
        print(f"      • Spatial size: {height}×{width}")
        print(f"      • Channels: {channels}")
        print(f"      • Chunk sizes: {[chunk['content'].shape for chunk in temporal_chunks]}")
        
        # Initialize result tensor with correct dimensions
        result = torch.zeros(
            (total_frames, height, width, channels),
            dtype=first_chunk_content.dtype,
            device=first_chunk_content.device
        )
        
        current_t = 0
        for i, chunk_info in enumerate(temporal_chunks):
            chunk_content = chunk_info['content']
            chunk_frames = chunk_content.shape[0]
            
            if i == 0:
                # First chunk - place entirely
                result[current_t:current_t+chunk_frames] = chunk_content
                current_t += chunk_frames
            else:
                # Subsequent chunks - use full available overlap for blending
                temporal_tile_info = temporal_tiles[i]
                expected_start = temporal_tile_info[0]
                overlap_frames = current_t - expected_start
                
                if overlap_frames > 0:
                    # Get regions for blending (using full overlap)
                    result_overlap = result[current_t-overlap_frames:current_t]
                    chunk_overlap = chunk_content[:overlap_frames]
                    
                    # Create temporal fade mask
                    fade_mask = self._create_temporal_fade_mask(overlap_frames, result_overlap.dtype, result_overlap.device)
                    
                    # Apply fade blending
                    blended_overlap = result_overlap * (1.0 - fade_mask) + chunk_overlap * fade_mask
                    result[current_t-overlap_frames:current_t] = blended_overlap
                    
                    # Place the non-overlapping part
                    if chunk_frames > overlap_frames:
                        result[current_t:current_t+chunk_frames-overlap_frames] = chunk_content[overlap_frames:]
                    current_t += chunk_frames - overlap_frames
                else:
                    # No overlap - place directly
                    result[current_t:current_t+chunk_frames] = chunk_content
                    current_t += chunk_frames
        
        return result
    
    def _create_temporal_fade_mask(self, fade_size, dtype, device):
        """Create a temporal fade mask for blending across time dimension."""
        fade_mask = torch.linspace(0, 1, fade_size, dtype=dtype, device=device)
        # Shape: (fade_size, 1, 1, 1) for broadcasting
        return fade_mask.view(fade_size, 1, 1, 1)
    
    def _force_memory_cleanup(self, model, vae):
        """Force memory cleanup between tile processing"""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Try to offload models if they have offload methods
            if hasattr(model, 'cpu_offload'):
                model.cpu_offload()
            elif hasattr(model, 'to'):
                model.to('cpu')
                
            if hasattr(vae, 'cpu_offload'):
                vae.cpu_offload()
            elif hasattr(vae, 'to'):
                vae.to('cpu')
                
        except Exception as cleanup_error:
            print(f"      ⚠️  Warning: Memory cleanup failed: {str(cleanup_error)}")
    
    def _calculate_temporal_tiles(self, total_frames, target_frames, overlap):
        """Calculate temporal tile ranges with overlap handling"""
        tiles = []
        
        if total_frames <= target_frames:
            tiles.append((0, total_frames))
        else:
            stride = target_frames - overlap
            current = 0
            
            while current < total_frames:
                end = min(current + target_frames, total_frames)
                remaining = total_frames - end
                
                if remaining > 0:
                    tiles.append((current, end))
                    if remaining < stride:
                        final_start = total_frames - target_frames
                        tiles.append((final_start, total_frames))
                        break
                else:
                    tiles.append((current, end))
                    break
                    
                current += stride
        
        return tiles
    
    def _calculate_spatial_tiles(self, total_size, target_size, overlap):
        """Calculate spatial tile ranges with overlap handling"""
        tiles = []
        
        if total_size <= target_size:
            tiles.append((0, total_size))
        else:
            stride = target_size - overlap
            current = 0
            
            while current < total_size:
                end = min(current + target_size, total_size)
                remaining = total_size - end
                
                if remaining > 0:
                    tiles.append((current, end))
                    if remaining < stride:
                        final_start = total_size - target_size
                        tiles.append((final_start, total_size))
                        break
                else:
                    tiles.append((current, end))
                    break
                    
                current += stride
        
        return tiles
    
    def _apply_temporal_overwriting(self, video_tile, mask_tile, temporal_idx, temporal_tiles, 
                                   frame_overlap, previous_chunk_stitched_frame, 
                                   t_start, h_start, h_end, w_start, w_end, debug_mode):
        """
        Apply temporal overwriting for consistency between temporal chunks.
        
        Returns:
            tuple: (modified_video_tile, modified_mask_tile)
        """
        if temporal_idx == 0 or previous_chunk_stitched_frame is None:
            return video_tile, mask_tile
        
        # Determine how many frames to overwrite
        frames_to_overwrite = frame_overlap
        if temporal_idx == len(temporal_tiles) - 1:
            # Last chunk: Look at previous chunk end frame
            previous_chunk_end_frame_index = temporal_tiles[temporal_idx - 1][1] - 1
            # Use as many frame as actual overlap allows
            frames_to_overwrite = previous_chunk_end_frame_index - t_start + 1
        
        # Extract corresponding frames from previous chunk's stitched result
        prev_chunk_frames = previous_chunk_stitched_frame.shape[0]
        source_start_idx = max(0, prev_chunk_frames - frames_to_overwrite)
        source_frames = previous_chunk_stitched_frame[source_start_idx:, h_start:h_end, w_start:w_end, :].clone()  # HARD COPY
        
        # Overwrite first frames of current tile with last frames from previous chunk
        if frames_to_overwrite > 0:
            video_tile[:frames_to_overwrite] = source_frames
            # Zero out corresponding mask areas
            mask_tile[:frames_to_overwrite] = 0.0
            
            if debug_mode:
                print(f"         🔄 Temporal overwrite: {frames_to_overwrite} frames from prev chunk")
        
        return video_tile, mask_tile
    
    def _apply_spatial_overwriting_left(self, video_tile, mask_tile, temporal_idx, h_idx, w_idx, 
                                       w_start, current_chunk_tiles, debug_mode):
        """
        Apply spatial overwriting from LEFT neighbor for consistency.
        
        Returns:
            tuple: (modified_video_tile, modified_mask_tile)
        """
        if w_idx == 0:
            return video_tile, mask_tile
        
        # Find left neighbor
        left_neighbor = None
        for tile in current_chunk_tiles:
            if (tile.temporal_index == temporal_idx and 
                tile.line_index == h_idx and 
                tile.column_index == w_idx - 1):
                left_neighbor = tile
                break
        
        if left_neighbor is not None and hasattr(left_neighbor, 'content'):
            # Calculate overlap region
            left_tile_end = left_neighbor.spatial_range_w[1]
            current_tile_start = w_start
            overlap_width = left_tile_end - current_tile_start
            
            if overlap_width > 0:
                # Extract overlap from left neighbor's right edge
                left_content = left_neighbor.content
                left_overlap = left_content[:, :, -overlap_width:, :]  # Last overlap_width columns
                
                # Overwrite left edge of current tile
                video_tile[:, :, :overlap_width, :] = left_overlap
                # Zero out corresponding mask areas
                mask_tile[:, :, :overlap_width] = 0.0
                
                if debug_mode:
                    print(f"         ↔️ Spatial overwrite LEFT: {overlap_width} pixels from neighbor")
        
        return video_tile, mask_tile
    
    def _apply_spatial_overwriting_top(self, video_tile, mask_tile, temporal_idx, h_idx, w_idx, 
                                      h_start, current_chunk_tiles, debug_mode):
        """
        Apply spatial overwriting from TOP neighbor for consistency.
        
        Returns:
            tuple: (modified_video_tile, modified_mask_tile)
        """
        if h_idx == 0:
            return video_tile, mask_tile
        
        # Find top neighbor
        top_neighbor = None
        for tile in current_chunk_tiles:
            if (tile.temporal_index == temporal_idx and 
                tile.line_index == h_idx - 1 and 
                tile.column_index == w_idx):
                top_neighbor = tile
                break
        
        if top_neighbor is not None and hasattr(top_neighbor, 'content'):
            # Calculate overlap region
            top_tile_end = top_neighbor.spatial_range_h[1]
            current_tile_start = h_start
            overlap_height = top_tile_end - current_tile_start
            
            if overlap_height > 0:
                # Extract overlap from top neighbor's bottom edge
                top_content = top_neighbor.content
                top_overlap = top_content[:, -overlap_height:, :, :]  # Last overlap_height rows
                
                # Overwrite top edge of current tile
                video_tile[:, :overlap_height, :, :] = top_overlap
                # Zero out corresponding mask areas
                mask_tile[:, :overlap_height, :] = 0.0
                
                if debug_mode:
                    print(f"         ↕️ Spatial overwrite TOP: {overlap_height} pixels from neighbor")
        
        return video_tile, mask_tile
    
    def _apply_spatial_overwriting_corner(self, video_tile, mask_tile, temporal_idx, h_idx, w_idx, 
                                         h_start, w_start, current_chunk_tiles, debug_mode):
        """
        Apply spatial overwriting from TOP-LEFT corner neighbor for consistency.
        
        Returns:
            tuple: (modified_video_tile, modified_mask_tile)
        """
        if h_idx == 0 or w_idx == 0:
            return video_tile, mask_tile
        
        # Find top-left diagonal neighbor
        topleft_neighbor = None
        for tile in current_chunk_tiles:
            if (tile.temporal_index == temporal_idx and 
                tile.line_index == h_idx - 1 and 
                tile.column_index == w_idx - 1):
                topleft_neighbor = tile
                break
        
        if topleft_neighbor is not None and hasattr(topleft_neighbor, 'content'):
            # Calculate corner overlap
            top_tile_end_h = topleft_neighbor.spatial_range_h[1]
            left_tile_end_w = topleft_neighbor.spatial_range_w[1]
            current_start_h = h_start
            current_start_w = w_start
            
            overlap_h = top_tile_end_h - current_start_h
            overlap_w = left_tile_end_w - current_start_w
            
            if overlap_h > 0 and overlap_w > 0:
                # Extract corner from top-left neighbor
                topleft_content = topleft_neighbor.content
                corner_overlap = topleft_content[:, -overlap_h:, -overlap_w:, :]
                
                # Overwrite corner of current tile
                video_tile[:, :overlap_h, :overlap_w, :] = corner_overlap
                # Zero out corresponding mask areas
                mask_tile[:, :overlap_h, :overlap_w] = 0.0
                
                if debug_mode:
                    print(f"         ↖️ Spatial overwrite CORNER: {overlap_h}x{overlap_w} from diagonal neighbor")
        
        return video_tile, mask_tile
    
    def _determine_tile_reference_images(self, temporal_idx, temporal_tiles, frame_overlap, 
                                        previous_chunk_stitched_frame, h_idx, w_idx, 
                                        debug_mode, kwargs):
        """
        Determine reference images for frame-wise temporal consistency.
        
        Returns:
            tile_ref_images: Reference images tensor or None
        """
        if temporal_idx == 0:
            # First temporal chunk: Use user-provided ref_images (if any)
            tile_ref_images = kwargs.get("vace_ref_images")
            if debug_mode and h_idx == 0 and w_idx == 0:  # Only print once per chunk
                print(f"      🎯 First chunk: Using user-provided ref_images")
            return tile_ref_images
        else:
            # Subsequent chunks: Use COMPLETE STITCHED FRAME from previous chunk as reference
            if previous_chunk_stitched_frame is not None:
                # Extract reference frame: frame_overlap frames before the last frame
                ref_frame_idx = previous_chunk_stitched_frame.shape[0] - frame_overlap - 1
                ref_frame_idx = max(0, ref_frame_idx)  # Ensure non-negative
                
                # Extract single COMPLETE FRAME as reference image
                tile_ref_images = previous_chunk_stitched_frame[ref_frame_idx:ref_frame_idx+1].clone()  # Shape: [1, H, W, C] - HARD COPY
                
                if debug_mode and h_idx == 0 and w_idx == 0:  # Only print once per chunk
                    print(f"      🔗 Frame-wise temporal chain: Using complete stitched frame {ref_frame_idx} from previous chunk T{temporal_idx-1}")
                    print(f"         Reference frame shape: {tile_ref_images.shape}")
                return tile_ref_images
            else:
                # Fallback: Use user-provided ref_images
                tile_ref_images = kwargs.get("vace_ref_images")
                if debug_mode and h_idx == 0 and w_idx == 0:
                    print(f"      ⚠️  No previous stitched frame found, using user ref_images as fallback")
                return tile_ref_images
    
    def _place_tile_with_overlap(self, target, tile, h_start, h_end, w_start, w_end, spatial_overlap):
        """Place tile into target tensor, handling overlaps with spatial fade blending"""
        tile_h, tile_w = tile.shape[1:3]
        
        # Get the target region
        target_region = target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :]
        
        # Check if there's existing content (non-zero) in the target area
        if target_region.sum() > 0:
            # Use production-quality fade blending
            fade_mask = self._create_spatial_fade_mask(tile_h, tile_w, target_region, tile, spatial_overlap)
            
            # Apply fade blending: existing * (1 - fade_mask) + tile * fade_mask
            blended = target_region * (1.0 - fade_mask) + tile * fade_mask
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = blended
        else:
            # First tile in this area - place directly
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = tile
    
    def _create_spatial_fade_mask(self, tile_h, tile_w, existing, new_tile, spatial_overlap):
        """Create a spatial fade mask for smooth blending between tiles"""
        # Create base mask (1.0 = use new tile, 0.0 = use existing)
        mask = torch.ones(1, tile_h, tile_w, 1, dtype=existing.dtype, device=existing.device)
        
        # Check which edges have existing content
        has_top = existing[:, :spatial_overlap, :, :].sum() > 0
        has_bottom = existing[:, -spatial_overlap:, :, :].sum() > 0
        has_left = existing[:, :, :spatial_overlap, :].sum() > 0
        has_right = existing[:, :, -spatial_overlap:, :].sum() > 0
        
        # Create fade gradients for each edge
        if has_top:
            for i in range(spatial_overlap):
                alpha = i / spatial_overlap
                mask[:, i, :, :] = alpha
        
        if has_bottom:
            for i in range(spatial_overlap):
                alpha = 1.0 - (i / spatial_overlap)
                mask[:, tile_h - 1 - i, :, :] = alpha
        
        if has_left:
            for i in range(spatial_overlap):
                alpha = i / spatial_overlap
                mask[:, :, i, :] = torch.minimum(mask[:, :, i, :], torch.tensor(alpha, dtype=mask.dtype, device=mask.device))
        
        if has_right:
            for i in range(spatial_overlap):
                alpha = 1.0 - (i / spatial_overlap)
                mask[:, :, tile_w - 1 - i, :] = torch.minimum(mask[:, :, tile_w - 1 - i, :], torch.tensor(alpha, dtype=mask.dtype, device=mask.device))
        
        return mask.expand_as(new_tile)
    
    def _stitch_temporal_chunks(self, chunks, temporal_tiles, total_frames, height, width, channels, overlap):
        """Stitch temporal chunks back together with temporal fade blending"""
        result = torch.zeros((total_frames, height, width, channels), dtype=chunks[0].dtype, device=chunks[0].device)
        
        for i, ((t_start, t_end), chunk) in enumerate(zip(temporal_tiles, chunks)):
            chunk_frames = chunk.shape[0]
            
            if i == 0:
                result[t_start:t_start+chunk_frames] = chunk
            else:
                prev_end = temporal_tiles[i-1][1]
                overlap_start = max(t_start, prev_end - overlap)
                overlap_end = min(t_end, prev_end)
                
                if overlap_start < overlap_end:
                    overlap_frames = overlap_end - overlap_start
                    chunk_overlap_start = overlap_start - t_start
                    
                    existing_frames = result[overlap_start:overlap_end]
                    new_frames = chunk[chunk_overlap_start:chunk_overlap_start+overlap_frames]
                    
                    fade_mask = self._create_temporal_fade_mask(overlap_frames, existing_frames.dtype, existing_frames.device)
                    
                    blended_frames = existing_frames * (1.0 - fade_mask) + new_frames * fade_mask
                    result[overlap_start:overlap_end] = blended_frames
                    
                    if overlap_end < t_start + chunk_frames:
                        non_overlap_start = overlap_end
                        chunk_offset = non_overlap_start - t_start
                        result[non_overlap_start:t_start+chunk_frames] = chunk[chunk_offset:]
                else:
                    result[t_start:t_start+chunk_frames] = chunk
        
        return result
    
    def _create_temporal_fade_mask(self, overlap_frames, dtype, device):
        """Create a temporal fade mask for smooth frame transitions"""
        fade_values = torch.linspace(0.0, 1.0, overlap_frames, dtype=dtype, device=device)
        fade_mask = fade_values.view(overlap_frames, 1, 1, 1)
        return fade_mask
    
    def _generate_wanvideo_processing_summary(self, all_tiles, temporal_tiles, spatial_tiles_h, spatial_tiles_w, total_tiles):
        """Generate comprehensive summary of the WanVideo tiled processing"""
        successful_tiles = sum(1 for tile in all_tiles if hasattr(tile, 'processing_status') and tile.processing_status == 'success')
        failed_tiles = total_tiles - successful_tiles
        
        summary = f"=== DIMENSION-WISE TILED WANVIDEO VACE PROCESSING SUMMARY ===\n"
        summary += f"Total tiles processed: {total_tiles}\n"
        summary += f"Successful WanVideo tiles: {successful_tiles}\n"
        summary += f"Failed tiles: {failed_tiles}\n"
        summary += f"WanVideo success rate: {(successful_tiles/total_tiles)*100:.1f}%\n\n"
        
        # Temporal consistency chain information
        summary += f"TEMPORAL CONSISTENCY CHAIN:\n"
        summary += f"Temporal chunks: {len(temporal_tiles)}\n"
        for i, (start, end) in enumerate(temporal_tiles):
            if i == 0:
                ref_source = "User-provided reference images"
            else:
                ref_source = f"Frame references from Chunk {i-1} (frame_overlap-based)"
            summary += f"  Chunk {i+1}: frames {start}-{end-1} ({end-start} frames) - {ref_source}\n"
        
        summary += f"\nSpatial tiles per frame: {len(spatial_tiles_h)}×{len(spatial_tiles_w)}\n"
        summary += f"Height tiles: {len(spatial_tiles_h)}\n"
        for i, (start, end) in enumerate(spatial_tiles_h):
            summary += f"  Row {i+1}: pixels {start}-{end-1} ({end-start} pixels)\n"
        
        summary += f"\nWidth tiles: {len(spatial_tiles_w)}\n"
        for i, (start, end) in enumerate(spatial_tiles_w):
            summary += f"  Col {i+1}: pixels {start}-{end-1} ({end-start} pixels)\n"
        
        # Analyze temporal consistency chain usage
        chain_usage = {}
        for tile in all_tiles:
            if hasattr(tile, 'ref_frame_source'):
                source = tile.ref_frame_source
                if source not in chain_usage:
                    chain_usage[source] = 0
                chain_usage[source] += 1
        
        summary += f"\nTemporal reference chain usage:\n"
        for source, count in chain_usage.items():
            summary += f"  {source}: {count} tiles\n"
        
        # Group tiles by processing status
        if failed_tiles > 0:
            summary += f"\nFailed tiles details:\n"
            for tile in all_tiles:
                if hasattr(tile, 'processing_status') and tile.processing_status != 'success':
                    summary += f"  T{tile.temporal_index}:L{tile.line_index}:C{tile.column_index}: {tile.processing_status}\n"
        
        # Add dimension-wise processing information
        summary += f"\nDimension-wise processing algorithm with temporal consistency:\n"
        summary += f"  1. Temporal consistency chain: Each chunk uses references from previous chunk\n"
        summary += f"  2. WanVideo VACE processing: Each tile processed with spatial-specific references\n"
        summary += f"  3. Column-wise stitching: Vertical fade blending within temporal chunks\n"
        summary += f"  4. Line-wise stitching: Horizontal fade blending across columns\n"
        summary += f"  5. Temporal stitching: Frame-based fade blending across time\n"
        summary += f"  6. Cropping: Final video cropped to original input dimensions\n"
        
        summary += f"\nTemporal consistency ensures each tile position maintains coherent evolution across chunks\n"
        summary += f"Reference frame extracted from frame_overlap position for smooth temporal transitions\n"
        summary += f"Large video processed through {total_tiles} individual WanVideo VACE operations with temporal continuity\n"
        
        return summary

    
