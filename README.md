# cuda-inference

## Phase 1: SIMT Microarchitecture & Scheduling
**Focus: The Instruction Pipeline**
- How the hardware fetches, schedules, and executes code at the Streaming Multiprocessor (SM) level.
- It focuses on the Warp Scheduler, instruction dispatch mechanics, and latency hiding.
* **Key Concepts:** SM Partitioning (SMSP), Warp Lifecycle (Dispatch/Stall), Register Pressure & Spilling, Predicated Execution (Branching), and Instruction Level Parallelism (ILP).

## Phase 2: Global Memory & Transaction Efficiency
**Focus: The Memory Controller (DRAM)**
- Optimization of the bus between VRAM and the SM.
- The goal is to maximize "Useful Bytes per Transaction" by aligning software access patterns with physical hardware constraints.
* **Key Concepts:** Sector-Based Loading (32-byte transactions), Coalescing Protocols, Vectorized Access (`LD.E.128`), and Memory Alignment strategies.

## Phase 3: The On-Chip Hierarchy (SRAM/L1)
**Focus: Data Locality & Reuse**
- Manual management of the software-controlled cache (Shared Memory) to minimize DRAM bandwidth usage.
- It addresses complex access patterns required for tensor operations.
* **Key Concepts:** Shared Memory Banking (32 banks), Conflict Resolution (Padding/Swizzling), Memory Consistency Models (Weak Ordering), and L2 Persistence.

## Phase 4: Warp Intrinsics (Register-Level Compute)
**Focus: Register-to-Register Communication**
- Low-latency reductions and voting mechanisms by bypassing the memory hierarchy entirely, allowing threads within a warp to exchange data directly via registers.
* **Key Concepts:** Shuffle Primitives (`__shfl_down_sync`, `__shfl_xor_sync`), Warp Voting (`__all_sync`, `__ballot_sync`), Lane Masking, and Sub-Warp Grouping.

## Phase 5: Asynchronous Pipelines (A100/Hopper)
**Focus: Hardware Latency Hiding**
- Saturating Tensor Cores and HBM bandwidth using modern hardware features that overlap data movement with computation (Double/Triple Buffering).
* **Key Concepts:** Asynchronous Copy (`cp.async`), Multi-Stage Pipelining, Hardware Barriers (`mbarrier`), Tensor Core APIs (`wmma`), and A100-specific Occupancy Control.
