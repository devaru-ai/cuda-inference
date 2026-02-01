# Phase 3: The Bitwise Mechanic

In Phase 1, we treated memory as bytes. In Phase 3, we treat memory as **bits**.
You need this for two reasons:

1. **Quantization:** Compressing data (e.g., packing four 8-bit integers into one 32-bit register).
2. **Banking:** Twisting memory addresses (Swizzling) to avoid hardware conflicts.


# 1. Bitwise Operators (The Hardware Tools)

**The Concept:**
ALUs (Arithmetic Logic Units) love bitwise math. It is often faster than integer math.

* **XOR (`^`):** The "Toggle" switch. Used to permute indices.
* **Shift (`<<`, `>>`):** Fast multiplication/division by 2.
* **And (`&`):** The "Mask." Isolating specific bits.

**XOR Swizzling**
This is the standard technique in libraries like **FlashAttention** to prevent Shared Memory Bank Conflicts.

* *Problem:* Accessing `tile[0]`, `tile[32]`, `tile[64]` hits Bank 0 repeatedly.
* *Fix:* You XOR the address with bits from the row index.
* *Result:* `tile[0]` hits Bank 0. `tile[32]` hits Bank 1. `tile[64]` hits Bank 2. (Randomized access).

**Code Snippet:**

```cpp
// 1. Fast Division/Mod (Power of 2 only)
int lane_id = threadIdx.x & 31; // Same as threadIdx.x % 32
int warp_id = threadIdx.x >> 5; // Same as threadIdx.x / 32

// 2. Swizzling Logic (The "Anti-Conflict" Pattern)
// Instead of accessing shared_mem[tid], we twist the index
int swizzled_idx = tid ^ (tid >> 5); 
float val = shared_mem[swizzled_idx];

```



# 2. Alignment (`alignas` / `__align__`)

**The Concept:**
The A100 memory controller fetches 32-byte sectors. If you tell it to load a 128-bit vector (`float4`) from an address like `0x004` (not divisible by 16), it fails.
You must force the compiler to pad your structures so they sit on clean memory boundaries.

**The Syntax:**

* `struct __align__(16) float4 { ... };` (CUDA style)
* `struct alignas(16) MyType { ... };` (C++11 style)

# Note

* *Question:* "I created a custom struct `MyPixel` with 3 floats (R, G, B). I made an array of them. Why is my `reinterpret_cast<float4*>` crashing?"
* *Answer:* `sizeof(MyPixel)` is 12 bytes.
* Element 0 starts at 0. (Aligned).
* Element 1 starts at 12. (**Misaligned** for 16-byte load).
* *Fix:* Add `alignas(16)`. The compiler will insert 4 bytes of invisible padding so every element starts on a 16-byte boundary.



**Code Snippet:**

```cpp
// BAD: Size is 12 bytes. Not friendly for vectorized loads.
struct Pixel {
    float r, g, b;
};

// GOOD: Size is 16 bytes. Hardware friendly.
struct alignas(16) AlignedPixel {
    float r, g, b;
    // Compiler secretly adds "float padding;" here
};

```


# 3. Unions (Safe Type Punning)

**The Concept:**
Sometimes `reinterpret_cast` is ugly or technically "Undefined Behavior" in pure C++. A `union` lets different data types occupy the exact same memory address.

**The Use Case: Half Precision (FP16)**
A `half2` (two FP16s) is stored as a single 32-bit integer in registers. Sometimes you need to perform bitwise hacks on the top 16 bits vs the bottom 16 bits without converting them to floats.

**Code Snippet:**

```cpp
union Packer {
    float f;
    int i;
};

// Goal: Check if a float is negative using ONLY integer math
// (Faster than float comparison on some ancient hardware, but good for understanding bits)
__device__ bool is_negative(float input) {
    Packer p;
    p.f = input;
    // Check the Sign Bit (Bit 31)
    return (p.i >> 31) & 1;
}

```


# 4. Summary: The "Packing" Kernel

To pass Phase 3, solve this quantization problem.

**Task:**
You have an array of 4 `float` values (32-bit each). You want to quantize them into 4 `int8` values and pack them into a single 32-bit `int` to write to global memory.

1. Cast float to int (simple cast).
2. Use shifts `<<` to move byte 0, 1, 2, 3 into position.
3. Use OR `|` to combine them.
4. Write result.

**Solution:**

```cpp
__global__ void pack_kernel(float* src, int* dst) {
    int tid = threadIdx.x;
    
    // 1. Load 4 floats
    float f0 = src[tid * 4 + 0];
    float f1 = src[tid * 4 + 1];
    float f2 = src[tid * 4 + 2];
    float f3 = src[tid * 4 + 3];
    
    // 2. Quantize (Simplified: just cast)
    unsigned int b0 = (unsigned int)f0; 
    unsigned int b1 = (unsigned int)f1;
    unsigned int b2 = (unsigned int)f2;
    unsigned int b3 = (unsigned int)f3;
    
    // 3. Bitwise Pack
    // Result: [ b3 | b2 | b1 | b0 ] (32 bits total)
    unsigned int packed = (b0 & 0xFF)       | 
                          ((b1 & 0xFF) << 8)  | 
                          ((b2 & 0xFF) << 16) | 
                          ((b3 & 0xFF) << 24);
                          
    // 4. Write 1 integer (saves 75% bandwidth vs writing 4 bytes separately)
    dst[tid] = packed;
}

```

**Checklist for success:**

* [ ] Did you mask with `& 0xFF`? (Safety: ensures you only grab the bottom 8 bits).
* [ ] Did you shift by 0, 8, 16, 24?
* [ ] Do you see how `sizeof(int)` (4 bytes) now holds 4 distinct variables?

