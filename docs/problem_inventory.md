# KernelBench-v3 Problem Inventory

## Summary

| Category | Count |
|----------|-------|
| Level 1 | 15 |
| Level 2 | 15 |
| Level 3 | 3 |
| Level 4 | 8 |
| Graphics | 2 |
| Tile Specialized | 13 |
| CuTile | 3 |
| Metal Level 1 (Image/Signal) | 8 |
| Metal Level 2 (Physics/Sim) | 6 |
| Metal Level 3 (Rendering) | 6 |
| Metal Level 4 (Scientific) | 6 |
| **Total** | **85** |

## Benchmark Mapping

- CUDA/Triton: problems/level1..level4
- CUTLASS/CuTe: problems/tile_specialized
- CuTile: problems/cutile
- Metal: problems/level1..level4 + problems/metal_level1..metal_level4 (excludes FP8, INT4, GatedDeltaNet, KDA)
- Graphics: problems/graphics

## CuTileBench Status

- Harness scripts implemented: `cutile_eval.py`, `cutile_batch_eval.py`, `src/prompts/cutile_system.py`
- Modal CUDA image: `nvidia/cuda:13.1.0-devel-ubuntu24.04`
- Runtime dependency:
  - installs `cuda-tile` (`cutile-python`) in Modal image
  - import path is `import cuda.tile as ct`
- Hardware support (current runtime):
  - B200 only (`tileiras` in CUDA 13.1 rejects `sm_90`)
- Validation:
  - Dry-run passed: `uv run python cutile_batch_eval.py --models minimax/minimax-m2.5 --gpus B200 --levels 1 --problems-per-level 1 --dry-run`
  - Real run status depends on model capability and CUDA/driver support for Tile IR on target GPU.
- Current state: CuTileBench is implemented as a separate harness and problem set using the official Python API.

## Per-Model Run Counts

| Benchmark | Problems | Runs (9 models) |
|-----------|----------|-----------------|
| CUDABench | 41 | 369 |
| TritonBench | 41 | 369 |
| CUTLASSBench | 13 | 117 |
| CuTeBench | 13 | 117 |
| CuTileBench | 3 | 27 |
| MetalBench | 63 | 567 |
| GraphicsBench | 2 | 18 |
| **Total** |  | **1386** |

## Detailed Listings

### Level 1
1. problems/level1/1_Square_matrix_multiplication_.py
2. problems/level1/23_Softmax.py
3. problems/level1/26_GELU_.py
4. problems/level1/2_Standard_matrix_multiplication_.py
5. problems/level1/36_RMSNorm_.py
6. problems/level1/3_Batched_matrix_multiplication.py
7. problems/level1/40_LayerNorm.py
8. problems/level1/42_Max_Pooling_2D.py
9. problems/level1/47_Sum_reduction_over_a_dimension.py
10. problems/level1/4_Matrix_vector_multiplication_.py
11. problems/level1/63_conv_standard_2D__square_input__square_kernel.py
12. problems/level1/82_conv_depthwise_2D_square_input_square_kernel.py
13. problems/level1/8_Matmul_with_irregular_shapes_.py
14. problems/level1/95_CrossEntropyLoss.py
15. problems/level1/9_Tall_skinny_matrix_multiplication_.py

### Level 2
1. problems/level2/17_Conv2d_InstanceNorm_Divide.py
2. problems/level2/37_Matmul_Swish_Sum_GroupNorm.py
3. problems/level2/40_Matmul_Scaling_ResidualAdd.py
4. problems/level2/46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py
5. problems/level2/52_Conv2d_Activation_BatchNorm.py
6. problems/level2/55_Matmul_MaxPool_Sum_Scale.py
7. problems/level2/59_Matmul_Swish_Scaling.py
8. problems/level2/66_Matmul_Dropout_Mean_Softmax.py
9. problems/level2/6_Conv3d_Softmax_MaxPool_MaxPool.py
10. problems/level2/73_Conv2d_BatchNorm_Scaling.py
11. problems/level2/82_Conv2d_Tanh_Scaling_BiasAdd_Max.py
12. problems/level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py
13. problems/level2/86_Matmul_Divide_GELU.py
14. problems/level2/98_Matmul_AvgPool_GELU_Scale_Max.py
15. problems/level2/99_Matmul_GELU_Softmax.py

### Level 3
1. problems/level3/31_VisionAttention.py
2. problems/level3/43_MinGPTCausalAttention.py
3. problems/level3/44_MiniGPTBlock.py

### Level 4
1. problems/level4/1_DeepSeek_MLA.py
2. problems/level4/2_DeepSeek_MoE.py
3. problems/level4/3_GroupedQueryAttention.py
4. problems/level4/4_FP8_Matmul.py
5. problems/level4/5_MoE_GatedGEMM.py
6. problems/level4/6_INT4_Quantized_GEMM.py
7. problems/level4/7_GatedDeltaNet.py
8. problems/level4/8_KimiDeltaAttention.py

### Graphics
1. problems/graphics/bloom.py
2. problems/graphics/particles.py

### Tile Specialized
1. problems/tile_specialized/gemm_bf16.py
2. problems/tile_specialized/gemm_bias_gelu.py
3. problems/tile_specialized/gemm_bias_relu.py
4. problems/tile_specialized/gemm_bias_silu.py
5. problems/tile_specialized/gemm_fp4.py
6. problems/tile_specialized/gemm_fp8.py
7. problems/tile_specialized/gemm_mixed_fp8_fp16.py
8. problems/tile_specialized/gemm_residual_add.py
9. problems/tile_specialized/gemv_bf16.py
10. problems/tile_specialized/gemv_fp16.py
11. problems/tile_specialized/gemv_fp4.py
12. problems/tile_specialized/gemv_fp8.py
13. problems/tile_specialized/moe_grouped_gemm.py

### CuTile
1. problems/cutile/persistent_gemm.py
2. problems/cutile/stream_k_gemm.py
3. problems/cutile/warp_specialized_gemm.py

### Metal Level 1 (Image/Signal Processing)
1. problems/metal_level1/gaussian_blur.py
2. problems/metal_level1/bilateral_filter.py
3. problems/metal_level1/histogram_equalization.py
4. problems/metal_level1/bicubic_resize.py
5. problems/metal_level1/sobel_edge_detect.py
6. problems/metal_level1/color_space_rgb_to_ycbcr.py
7. problems/metal_level1/fft_2d.py
8. problems/metal_level1/alpha_compositing.py

### Metal Level 2 (Physics/Simulation)
1. problems/metal_level2/particle_system.py
2. problems/metal_level2/nbody_gravity.py
3. problems/metal_level2/sph_density.py
4. problems/metal_level2/cloth_verlet.py
5. problems/metal_level2/collision_broadphase.py
6. problems/metal_level2/heat_diffusion_2d.py

### Metal Level 3 (Rendering/Graphics)
1. problems/metal_level3/bloom_effect.py
2. problems/metal_level3/tone_mapping_aces.py
3. problems/metal_level3/ray_sphere_intersection.py
4. problems/metal_level3/ray_triangle_moller.py
5. problems/metal_level3/ssao.py
6. problems/metal_level3/sdf_raymarching.py

### Metal Level 4 (Scientific Compute/Algorithms)
1. problems/metal_level4/prefix_sum.py
2. problems/metal_level4/radix_sort.py
3. problems/metal_level4/sparse_matvec_csr.py
4. problems/metal_level4/knn_points.py
5. problems/metal_level4/bitonic_sort.py
6. problems/metal_level4/monte_carlo_pi.py
