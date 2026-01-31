# SIMT Microarchitecture & Scheduling

- Understand exactly how the hardware maps your code to physical transistors and why "Latency Hiding" is the primary mechanic of GPU performance.

## The Physical Hierarchy
* **The SM (Streaming Multiprocessor):** The core unit. Contains 4 SMSPs (SM Sub-Partitions).
* **The SMSP (Sub-Core):** Contains 1 Warp Scheduler + 1 Dispatch Unit. It manages a fixed pool of warps (e.g., 16 slots).
* **The Register File:** The fastest memory, but the hardest constraint (64K 32-bit registers per SM).


## The Scheduling Loop
* Every cycle, the Warp Scheduler looks at its pool of "Active Warps."
* It checks which warps are **"Eligible"** (not stalled on Memory, Math, or Barriers).
* It picks **one** eligible warp and issues an instruction.
* *Key Insight:* If all warps are stalled (waiting for memory), the hardware is idle. This is why we need **Occupancy** (enough active warps to hide the wait).


# Implementation Projects

## Project 1.1: The Register Pressure Lab

* **File:** `register_pressure.cu`
* **Goal:** Force the compiler to "spill" registers and observe the crash in occupancy.
* **Task:**
1. Write a kernel with a heavy math loop using 2-3 local variables.
2. Write a second version `__launch_bounds__(max_threads, min_blocks)` or use volatile arrays to force high register usage (e.g., declare `float reg[100]`).
3. Use `cudaOccupancyMaxPotentialBlockSize` API to print the theoretical max occupancy for both.


* **Success Indicator:** You see the "Active Warps per SM" drop as you increase register usage per thread.

## Project 1.2: The Divergence Benchmark

* **File:** `warp_divergence.cu`
* **Goal:** Measure the precise cycle penalty of `if/else`.
* **Task:**
1. Kernel A (Coherent): `if (tid < 32) { ... }` (All threads take True path).
2. Kernel B (Divergent): `if (tid % 2 == 0) { ... } else { ... }` (Even/Odd split).
3. Use `clock()` or `__nanosleep()` inside the branches to simulate work.


* **Success Indicator:** Kernel B should run roughly 2x slower than Kernel A, proving that the hardware serializes the two paths.

## Project 1.3: Latency Hiding via ILP (Instruction Level Parallelism)

* **File:** `ilp_unrolling.cu`
* **Goal:** Hide latency *without* adding more threads.
* **Task:**
1. Kernel A (Serial): Load `x`, Compute `x^2`, Store `x`. (Dependency chain).
2. Kernel B (ILP): Load `x, y, z, w`. Compute `x^2, y^2...`. Store `x, y...`. (Independent instructions).
3. Run both with **1 Block, 1 Warp** (Lowest possible occupancy).


* **Success Indicator:** Kernel B achieves higher throughput despite low occupancy because the scheduler can issue the load for `y` while `x` is pending.

# Profiling & Analysis 

Use **Nsight Compute (`ncu`)** to validate your code. Do not guess; measure.

### Metric 1: `sm__warps_active.avg.pct_of_peak_sustained_active
- This is your **Occupancy**. Is it 80%? 20%? Why?


### Metric 2: `smsp__issue_inst0.avg.pct_of_peak_sustained_active
- This is your **Scheduler Efficiency**. If this is low, your warps are stalled.


### Metric 3: `smsp__warp_issue_stalled_barrier_per_warp_active.pct
- Are you waiting on `__syncthreads()`?

### Metric 4: `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
- Are you waiting on Global Memory?

# Can you Answer?

1. **The Scheduling Question:** "If an SM has 4 schedulers, can it issue instructions from the same warp twice in one cycle?"
* *Answer:* No. (Usually). A warp is assigned to *one* SMSP. It can generally issue 1 (or 2 dual-issue) instructions per cycle, but limited by the dispatch unit.


2. **The Divergence Question:** "Does `if (threadIdx.x < 16)` cause divergence in a warp?"
* *Answer:* No. Threads 0-15 are in the same warp? Yes. But wait—Warps are usually 32 threads. If the warp contains threads 0-31, and only 0-15 execute, the other 16 are masked off. It is technically divergent flow, but since the "Else" block is empty, the penalty is minimal (just the mask setting).


3. **The Latency Question:** "Why does low occupancy hurt performance on memory-bound kernels but matter less on compute-bound kernels (with high ILP)?"
* *Answer:* Memory latency is ~400 cycles. You need many other warps to do work while one waits. If you are compute-bound with high ILP, a single warp can keep the ALU busy without needing to switch to others.
