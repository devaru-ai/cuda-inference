# Phase 4: Warp Intrinsics

Stop thinking about "threads sharing memory." Think about "threads talking to each other."

# The Register Bypass
* Normally, Thread A communicates with Thread B by writing to Shared Memory (SRAM).
* **Warp Shuffle (`__shfl`)** allows Thread A to read a register from Thread B *directly* over the warp's internal data path.
* **Latency:** ~1-2 cycles (Shuffle) vs. ~20-30 cycles (Shared Memory Read/Write + Sync).


# The Primitives
* `__shfl_sync(mask, var, srcLane)`: Broadcast "I want `var` from thread `srcLane`."
* `__shfl_down_sync(mask, var, delta)`: "Give me `var` from the neighbor `delta` spots to the right."
* `__shfl_xor_sync(mask, var, mask)`: Butterfly pattern (used for bitwise reduction algorithms).


# The Active Mask (`0xffffffff`)
* Modern CUDA requires you to specify *which* threads are participating.
* *Trap:* If you use `0xffffffff` (all threads) inside a divergent `if` branch, you will get undefined behavior or hangs.


# 2. Implementation Projects 

### Project 4.1: The Register Reduction 

* **File:** `warp_reduce.cu`
* **Goal:** Sum 32 integers without touching memory.
* **Task:**
1. Kernel A (Shared Mem): Write to `s_data[tid]`, `__syncthreads()`, reduce in shared mem.
2. Kernel B (Shuffle): Use the "Log Step" pattern:
```cpp
int val = input[tid];
val += __shfl_down_sync(0xffffffff, val, 16);
val += __shfl_down_sync(0xffffffff, val, 8);
val += __shfl_down_sync(0xffffffff, val, 4);
val += __shfl_down_sync(0xffffffff, val, 2);
val += __shfl_down_sync(0xffffffff, val, 1);
// Thread 0 now holds the sum of all 32 threads.

```




* **Result:** The Shuffle kernel should be 10x+ faster for small reductions because it removes all `LDS/STS` (Load/Store Shared) instructions.

### Project 4.2: Warp Voting 

* **File:** `warp_vote.cu`
* **Goal:** Make decisions as a group.
* **Task:**
1. Each thread checks a condition (e.g., `data[tid] > threshold`).
2. Use `__ballot_sync(0xffffffff, predicate)` to get a 32-bit integer bitmask of results.
3. Use `__popc(mask)` (Population Count) to count how many threads passed.
4. *Application:* This is how you implement **Sparse** kernels (knowing how many non-zeros exist in the warp to allocate space).



### Project 4.3: The Block Reduction 

* **File:** `block_reduce.cu`
* **Goal:** Sum 1024 integers (1 Block).
* **Task:**
1. **Phase 1:** Each Warp reduces its own 32 elements using Shuffles (Project 4.1).
2. **Phase 2:** The first thread of each warp (Lane 0) writes the partial sum to Shared Memory.
3. **Phase 3:** `__syncthreads()`.
4. **Phase 4:** The first warp reads the partial sums and does one final Shuffle reduction.


* *Why:* This is the standard implementation in libraries like CUB.

#  Profiling & Analysis (Nsight Compute)**

You are looking for the absence of memory instructions.

* **SASS View (Assembly):**
* **Shared Mem Version:** You will see lots of `LDS` (Load Shared) and `STS` (Store Shared).
* **Shuffle Version:** You should see `SHFL.DOWN` instructions and almost zero `LDS/STS`.


* **Metric:** `smsp__inst_executed.avg`
* The Shuffle version should execute significantly fewer instructions to achieve the same math.



# Can you answer?

1. **The Sync Question:** "Why don't I need `__syncthreads()` between the steps of a Shuffle reduction?"
* *Answer:* Shuffle instructions are synchronous *within the warp* at the hardware level. The instruction itself guarantees that the data is ready before the next instruction executes (provided the mask is correct).


2. **The Mask Question:** "What happens if I use `__shfl_down_sync(0xffffffff, val, 1)` inside an `if (tid < 16)` block?"
* *Answer:* **Undefined Behavior / Deadlock.** You promised the hardware (via `0xffffffff`) that all 32 threads would participate, but 16 of them are masked off by the `if`. The participating threads wait forever for data from the inactive threads.


3. **The Performance Question:** "For a reduction of 1024 elements, where is the bottleneck? Memory or Compute?"
* *Answer:* It is almost always **Memory Bound** (loading 1024 elements from global memory). The reduction arithmetic (the shuffles) is so fast it is effectively free. The optimization focus should be on *loading* the data efficiently (Phase 2), not just the math.



