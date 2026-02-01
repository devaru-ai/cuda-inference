# Phase 2: The Compile-Time Optimizer (Template Metaprogramming)

In standard software, we use variables to make code flexible. In High-Performance Computing (HPC), **variables are slow**. Reading a variable takes registers and cycles.

The goal of Phase 2 is to move decision-making from **Runtime** (when the kernel is running) to **Compile Time** (when `nvcc` is building the binary). This allows the compiler to hard-code constants and delete dead branches.

# 1. Non-Type Template Parameters 

**The Concept:**
Instead of passing arguments like `block_size` or `is_training` to your function at runtime, you pass them as template arguments. This effectively creates a unique, hard-coded copy of the function for those specific values.

**Why:**
If the compiler knows `BLOCK_SIZE` is exactly 256, it can perform **strength reduction** (replacing expensive division `idx / 256` with cheap bitwise shifts `idx >> 8`). It cannot do this if 256 is a variable passed at runtime.

**The Syntax:**
`template <int VAL>` instead of `template <typename T>`.

**Code Snippet:**

```cpp
// SLOW (Runtime): Compiler treats 'width' as a variable
__global__ void matmul_dynamic(float* a, int width) {
    int idx = threadIdx.x + blockIdx.x * width; // Requires integer multiplication instruction
    // ...
}

// FAST (Compile Time): Compiler treats 'WIDTH' as a literal constant
template <int WIDTH>
__global__ void matmul_static(float* a) {
    int idx = threadIdx.x + blockIdx.x * WIDTH; 
    // Optimization: If WIDTH is 1024, compiler replaces multiply with '<< 10'
}

// Launching it requires the brackets:
matmul_static<1024><<<grid, block>>>(ptr);

```

# Note
* *Question:* "Why not just pass `width` as a `const int` argument?"
* *Answer:* Even `const` arguments are runtime values to the GPU (they live in Constant Memory or Registers). Only Template Parameters are truly "immediate values" in the assembly code (SASS).

# 2. `if constexpr` (Static Branching)

**The Concept:**
A normal `if` statement exists in the final binary. The hardware has to check the condition every time.
`if constexpr` (C++17) evaluates the condition during compilation. If the condition is False, the code block is **completely deleted** from the binary. It consumes zero instruction cache and zero registers.

**Why:**
This is critical for generic kernels that handle multiple data types or layouts. You can write one "Master Kernel" but compile optimized variants.

**Code Snippet:**

```cpp
template <bool USE_BIAS>
__device__ void dense_layer(float* out, float* bias, int idx) {
    // Math logic...
    
    // Runtime 'if': The branch instruction exists in SASS. 
    // GPU might predict it, but it takes space.
    if (USE_BIAS) { 
        out[idx] += bias[idx];
    }
    
    // Compile-time 'if constexpr':
    // If USE_BIAS is false, this line effectively ceases to exist.
    if constexpr (USE_BIAS) {
        out[idx] += bias[idx];
    }
}

```

# 3. Loop Unrolling (`#pragma unroll`)

**The Concept:**
Loops have overhead: incrementing the counter (`i++`), comparing (`i < N`), and jumping back. "Unrolling" means pasting the loop body  times linearly to remove that overhead.

**Why:**
Crucial for short loops inside a kernel (e.g., iterating over a small tile in Shared Memory). It also allows the compiler to reorder instructions (ILP) across iterations.

**The Syntax:**

* `#pragma unroll`: Unrolls fully (if compile knows the count).
* `#pragma unroll 4`: Unrolls 4 iterations at a time.

**Code Snippet:**

```cpp
template <int TILE_SIZE>
__device__ void load_tile(float* smem, float* gmem) {
    // Without unroll: Hardware jumps back and forth 4 times.
    // With unroll: Compiler pastes the load instruction 4 times sequentially.
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
        smem[threadIdx.x + i] = gmem[threadIdx.x + i];
    }
}

```

*Note: This only works if `TILE_SIZE` is a template parameter (Phase 2.1)! If `TILE_SIZE` is a runtime variable, the compiler cannot unroll it.*


# 4. Summary: The "Template Config" Kernel

To pass Phase 2, write this snippet (mentally or scratchpad).

**Task:**
Write a kernel `process_data` that takes an input array.

1. It is templated on an integer `SCALE_FACTOR`.
2. It is templated on a boolean `APPLY_RELU`.
3. It multiplies data by `SCALE_FACTOR`.
4. If `APPLY_RELU` is true, it sets negatives to zero using `if constexpr`.
5. Show how you would launch it for Scale=2, ReLU=True.

**Solution:**

```cpp
template <int SCALE_FACTOR, bool APPLY_RELU>
__global__ void process_data(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float val = data[idx];
        
        // Compiler hardcodes this multiply (likely to a shift if power of 2)
        val *= SCALE_FACTOR; 
        
        // If false, this block vanishes from binary
        if constexpr (APPLY_RELU) {
            if (val < 0.0f) val = 0.0f;
        }
        
        data[idx] = val;
    }
}

// Host Launch
// This generates a specific kernel named process_data_2_true in the symbol table
process_data<2, true><<<grid, block>>>(dev_ptr, 1024);

```

**Checklist for success:**

* [ ] Did you put `<int, bool>` in the `template` line?
* [ ] Did you realize that if you launch `<3, true>`, the compiler generates a *separate* kernel than `<2, true>`? (This is called Template Instantiation).

