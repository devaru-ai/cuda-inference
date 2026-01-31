# Phase 3: The On-Chip Hierarchy (SRAM/L1).

Only optimized kernels manually manage the L1 cache (Shared Memory) to achieve **Data Reuse**.

Shared Memory is not just a "fast array." It has a specific physical structure that you can accidentally break.

# The Banking System
* Shared Memory is divided into **32 Banks**.
* Each bank is **4 bytes** wide (32 bits).
* **Mapping Formula:** `Bank Index = (Address / 4) % 32`.
* *Implication:* Sequential `float` words map to sequential banks (0, 1, 2... 31). This is perfect for a warp reading a contiguous line.


# The Conflict (The Bottleneck)
* **N-Way Bank Conflict:** If multiple threads in a warp try to access *different addresses* that map to the *same bank* in the same cycle, the hardware **serializes** them.
* **Worst Case:** A 32-way conflict (e.g., accessing a column in a row-major array with stride 32). The instruction takes 32x longer to execute.


# The Exception (Broadcast)
* If multiple threads read the **exact same address**, the hardware "broadcasts" the value in 1 cycle. No conflict.


# 2. Implementation Projects

### Project 3.1: The Bank Conflict Laboratory

* **File:** `bank_conflicts.cu`
* **Goal:** Visualize the serialization penalty.
* **Task:**
1. Declare `__shared__ float data[1024]`.
2. **Kernel A (No Conflict):** `val = data[tid]` (Stride 1).
3. **Kernel B (Conflict):** `val = data[tid * 32]` (Stride 32).
4. **Kernel C (Padding):** Declare `data[32][33]` (2D array with +1 padding). Access column-wise.


* **Result:** Kernel B should be significantly slower. Kernel C should fix the column-access slowness by shifting the data layout so columns don't align to the same bank.

### Project 3.2: Tiled Matrix Multiplication 

* **File:** `tiled_matmul.cu`
* **Goal:** Implement the "Tiling" pattern to reduce Global Memory bandwidth.
* **Task:**
1. Define a Tile Size (e.g., ).
2. **Phase 1 (Load):** Threads cooperatively load a tile of Matrix A and B from Global  Shared Memory.
3. **Phase 2 (Sync):** `__syncthreads()`.
4. **Phase 3 (Compute):** Threads compute the dot product using *only* the data in Shared Memory.
5. **Phase 4 (Repeat):** Move to the next tile.


* **Success Metric:** You should verify that for , you perform ~32x fewer Global Memory loads compared to a naive kernel.

### Project 3.3: XOR Swizzling (The A100 Standard)

* **File:** `swizzle_demo.cu`
* **Goal:** Avoid conflicts without wasting memory on padding.
* **Task:**
1. Implement a mapping function using XOR: `int swizzled_idx = tid ^ (tid >> 5);`
2. Use this index to write/read from Shared Memory.
3. *Why:* This is how libraries like **CUTLASS** and **FlashAttention** store tensor tiles to ensure that row *and* column accesses are both conflict-free.


# Profiling & Analysis (Nsight Compute)

You need to verify that your padding/swizzling actually worked.

#### Metric 1: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
* The total number of conflicts.
* **Goal:** In Project 3.1 (Kernel C) and Project 3.2, this number should be **Zero** (or very close to it).


#### Metric 2: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
* Compare the Naive MatMul vs. Tiled MatMul. The Tiled version should have drastically fewer global sectors loaded.


# Can you answer?

1. **The Padding Question:** "I have a `__shared__ float tile[32][32]`. Threads read down the columns (`tile[row][tid]`). Why does this cause a 32-way bank conflict? How does changing it to `tile[32][33]` fix it?"
* *Answer:* With `[32][32]`, column elements (indices 0, 32, 64...) all map to Bank 0 (because  bytes is a multiple of the total bank width). Changing width to 33 shifts the stride so column element  lands in Bank .


2. **The Tiling Math Question:** "In a standard Matrix Multiplication of size , what is the Arithmetic Intensity (FLOPS / Bytes) of the naive version vs. the Tiled version (Tile width )?"
* *Answer:*
* Naive: Intensity  (Requires reading 2 floats for every FMA).
* Tiled: Intensity . (We reuse the data  times). This is why Tiling pushes us from "Memory Bound" towards "Compute Bound."




3. **The Sync Question:** "In the Tiled MatMul loop, what happens if you forget the second `__syncthreads()` after the computation but before the next load?"
* *Answer:* **Race Condition (Data Hazard).** Some threads might start loading the *new* tile (overwriting Shared Memory) while slower threads are still computing using the *old* tile.



