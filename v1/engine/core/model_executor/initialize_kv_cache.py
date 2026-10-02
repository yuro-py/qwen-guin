# A. Get per-layer KV-cache spec, cuz of hybrid architectures
# B. Run a dummy forward pass to take a snapshot of the gpu memory to compute the number of blocks that can be fitted in the available vram:-
# 2 * BLOCK_LENGTH * KV_HEADS * HIDDEN_DIM*  DTYPE_BYTES
# C. Allocate, reshape and bind KV cache tensors to attention layers
# D. SET ATTENTION METADATA WHICH ARE LATER REFERRED BY THE KERNELS DURING FORWARD PASS
# E. CUDA GRAPHS CAPTURE THE UPCOMING GPU WORK IN DAG IN A DUMMY FORWARD PASS SO THE ACTUAL FORWARD PASS CUTS KERNEL LAUNCH OVERHEAD AND REDUCE LATENCY. CAN BE TURNED OFF WITH "--ENFORCE-EAGER".
