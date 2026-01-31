# Phase 2: Global Memory & Transaction Efficiency
The invisible data bus between the GPU and the VRAM.

# The Transaction Unit (The 32-Byte Rule)
* The L2 Cache and Memory Controller do not speak in single bytes. The smallest unit they can move is a **32-byte sector**.
* **The Trap:** If a thread reads **4 bytes** (a float), the hardware fetches **32 bytes**.
* **The Efficiency Ratio:**
* If 32 threads read 32 sequential floats (128 bytes total), the hardware fetches four 32-byte sectors. Efficiency = 100%.
* If 32 threads read strided floats (stride 32), the hardware fetches 32 separate sectors (1024 bytes) to deliver just 128 bytes of useful data. Efficiency = 12.5%.


# Vectorized Loads (`LD.E.128`)
* The Load/Store Unit (LSU) is a pipeline. Issuing an instruction costs cycles.
* Loading 4 floats individually = 4 instructions + 4 address calculations.
* Loading 1 `float4` = 1 instruction + 1 address calculation. This increases **Instruction Level Parallelism (ILP)** by freeing up the pipeline.

# 2. Implementation Projects 

## Project 2.1: The Coalescing Benchmark

* **File:** `coalescing_stride.cu`
* **Goal:** Quantify the penalty of "Strided Access."
* **Task:**
1. Kernel A (Coalesced): `output[idx] = input[idx] * 2.0f;`
2. Kernel B (Strided): `output[idx * stride] = input[idx * stride] * 2.0f;`
3. Run Kernel B with `stride = 2`, `stride = 8`, `stride = 32`.


* **Prediction:** The execution time will not scale linearly. `Stride=32` will be roughly 8-10x slower, not 32x, because you saturate the memory controller's request queue.

## Project 2.2: The Misalignment Demo

* **File:** `misaligned_access.cu`
* **Goal:** Understand why `malloc` always returns aligned pointers.
* **Task:**
1. Allocate a generic `char*` buffer. Cast it to `float*`.
2. Kernel A: Read starting from byte 0. (Aligned).
3. Kernel B: Read starting from byte 4 (Aligned for float).
4. Kernel C: Read starting from byte 1 (Misaligned).


* **Why this matters:** On older architectures, C crashes. On modern architectures, C causes **2x transactions** (because the 4-byte float now straddles two 32-byte sectors).

## Project 2.3: Vectorization (The Bandwidth Maximizer)

* **File:** `vectorized_copy.cu`
* **Goal:** Saturate the bus using `float4`.
* **Task:**
1. Kernel A: Copy N floats using `float*`.
2. Kernel B: Cast pointers to `float4*` and copy N/4 elements.
3. Check the generated SASS (Assembly) using `cuobjdump -sass`. You want to see `LDG.E.128` instead of `LDG.E`.

# 3. Profiling & Analysis (Nsight Compute)

This is where you see the "invisible" traffic.

### Metric 1: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
* The total number of 32-byte sectors moved.
* *Check:* In the Strided kernel, is this number 32x higher than the Coalesced kernel?


### Metric 2: `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`
* This is your **"Sector Utilization"**.
* **100%:** Ideal (Coalesced).
* **3.125%:** Disaster (Stride 32). This proves you are wasting 97% of your bandwidth.


## Metric 3: `dram__throughput.avg.pct_of_peak_sustained_elapsed`
* Are you actually hitting the 1.5 TB/s limit of the A100? If your kernel is "Bandwidth Bound" but this number is low (e.g., 20%), your access pattern is the bottleneck.



# Can you answer?

1. **The Transaction Question:** "I have a warp of 32 threads. Each thread reads a single byte `char x = data[tid]`. How many 32-byte sectors are transferred from DRAM?"
* *Answer:* **One.** (If aligned). The 32 bytes form a perfect, contiguous 32-byte block. The hardware is smart enough to coalesce byte-level reads.


2. **The Stride Question:** "Why is `Stride = 2` (accessing every other float) bad? Draw the cache line."
* *Answer:* You fetch a 32-byte sector (8 floats). You only use floats 0, 2, 4, 6. You use 16 bytes but paid for 32. **50% efficiency.**


3. **The Vectorization Question:** "Why does using `float4` improve performance even if I have enough threads to saturate bandwidth with regular `float`?"
* *Answer:* It reduces **Instruction Overhead**. You issue 1/4th the number of instructions, reducing pressure on the Dispatch Unit and Warp Scheduler, leaving more cycles for math or hiding latency.

