# Phase 5: Asynchronous Pipelines (A100/Hopper)

The A100 is so fast that standard CUDA kernels cannot keep up. The math units (Tensor Cores) will sit idle waiting for data unless you use specialized hardware paths.

# The Problem (Latency)
* Even with Tiling (Phase 3), a standard `LDG` (Global Load) blocks a register. The thread cannot safely use that register until the data arrives (~400 cycles later).
* To hide this, you need massive occupancy. But A100 kernels often use *low* occupancy (large tiles).


# The Solution (`cp.async`)
* **Bypassing Registers:** `cp.async` tells the DMA engine: "Move data from Global directly to Shared Memory. Don't put it in my registers. I don't want to know about it."
* **The Benefit:** The CUDA threads are free to execute math instructions immediately after issuing the copy command.


# Double Buffering
* You divide Shared Memory into two buffers: A and B.
* **Loop:**
1. Compute on Buffer A.
2. *Simultaneously* issue `cp.async` to fill Buffer B.
3. Swap.




# Tensor Cores (`wmma`)
* Specialized ALUs that perform a  matrix multiply in one instruction.
* **Layouts:** The data in registers is opaque (fragmented). You cannot access `reg[0]` directly; you must use `load_matrix_sync` APIs.


# 2. Implementation Projects 


### Project 5.1: The Async Copy Pipeline

* **File:** `async_pipeline.cu`
* **Goal:** Implement a 2-stage pipeline manually.
* **Task:**
1. **Prologue:** Issue `cp.async` for Tile 0. `cp.async.commit_group()`. `cp.async.wait_group(0)`.
2. **Loop:**
* Issue `cp.async` for Tile . `cp.async.commit_group()`.
* **Compute** on Tile  (while the copy happens in background).
* `cp.async.wait_group(0)` (Wait for Tile  to finish before swapping).
* `__syncthreads()`.




* **Metric:** This should yield higher throughput than Project 3.2 (Tiled MatMul) on an A100, even with fewer warps.

### Project 5.2: Tensor Core "Hello World"

* **File:** `tensor_core_gemm.cu`
* **Goal:** Use the `nvcuda::wmma` API.
* **Task:**
1. Include `<mma.h>`.
2. Declare fragments: `wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;`
3. Load data: `wmma::load_matrix_sync(a_frag, ...);`
4. Compute: `wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);`
5. Store: `wmma::store_matrix_sync(..., c_frag, ...);`


* **Constraint:** You must use `__half` (FP16) data types. A100 Tensor Cores do not support FP32 inputs (only FP32 accumulation).

### Project 5.3: FlashAttention V1 

* **File:** `softmax_online.cu`
* **Goal:** Combine Phase 4 and Phase 5.
* **Task:**
* Implement the "Online Softmax" logic inside a kernel.
* Update `max` and `sum` incrementally as you load blocks of data.
* *Why:* This logic is the "Inference" part of the FlashAttention paper. It proves you can calculate Softmax without materializing the full  matrix in HBM.



# 3. Profiling & Analysis (Nsight Compute)

You are looking for **Async Activity**.

* **Metric 1: `pipe_lsu_mem_global_op_ld.async**`
* Are your loads actually async? If this is zero, you are using standard loads.


* **Metric 2: `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active**`
* **Tensor Core Utilization.** Are you keeping the beast fed? If this is low (<50%), your pipeline bubbles (the compute finished before the data arrived).
* *Fix:* Increase the pipeline depth (Triple Buffering) or optimize the copy size.


# Can you answer?

1. **The Pipeline Question:** "In a double-buffered pipeline, why do we need `cp.async.wait_group(N)`? What does `N` represent?"
* *Answer:* It tells the hardware to sleep until only `N` commit groups are pending. For double buffering, we usually commit the load for the *next* iteration, then calculate current. Before swapping, we wait until `N=0` (everything is done) or manage a rotating buffer where we wait for the specific group we need next.


2. **The Latency Question:** "Why does using `cp.async` allow us to run with *lower* occupancy (fewer warps)?"
* *Answer:* In standard CUDA, we need many warps to hide memory latency (Context Switching). With `cp.async`, the latency is hidden by the **overlap** of Copy and Compute within the *same* warp. We don't need to switch warps to find work; the current warp has work (math) to do while the copy engine works.


3. **The Layout Question:** "Why must Tensor Core inputs be in Shared Memory or Registers? Can I feed them directly from Global Memory?"
* *Answer:* You cannot feed them from Global. You must load to Registers (via `load_matrix_sync`) typically from Shared Memory (to handle the complex swizzling/layouts required by the hardware).


