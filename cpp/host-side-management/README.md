# Phase 4: Host-Side Management

You have written the perfect kernel. Now you need to launch it from the CPU (Host) without crashing your machine or leaking 80GB of VRAM.
Systems C++ is about **Safety** and **Abstraction**.

# 1. RAII (Resource Acquisition Is Initialization)

**The Concept:**
In C++, if you `malloc` (or `cudaMalloc`), you must `free`. If your program crashes or returns early *before* the free, that memory is lost until you reboot the process.
**RAII** means wrapping the pointer in a class.

* **Constructor:** Allocates memory.
* **Destructor:** Frees memory.
* **Result:** When the class goes out of scope (even via crash/exception), the destructor runs automatically. VRAM is safe.

**Code Snippet:**

```cpp
// The "Smart Pointer" for GPU memory
template <typename T>
struct DeviceBuffer {
    T* ptr;
    size_t size;

    DeviceBuffer(size_t n) : size(n) {
        cudaMalloc(&ptr, n * sizeof(T));
    }

    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }

    // Implicit cast to raw pointer for easy kernel launches
    operator T*() { return ptr; }
};

// Usage:
void run_model() {
    DeviceBuffer<float> buffer(1024); // Allocates
    my_kernel<<<1, 1>>>(buffer);      // Works like a pointer
    // Function ends -> Destructor runs -> cudaFree called automatically
}

```
# 2. Functors (Passing Logic to Kernels)

**The Concept:**
How do you write a generic `TransformKernel` that can do addition *or* multiplication?
Passing function pointers in CUDA is messy (device pointers).
Instead, we pass a **Functor**: A C++ struct that acts like a function (via `operator()`).

**The Syntax:**

1. Define a struct with `__device__ operator()`.
2. Pass the struct *by value* to the kernel template.

**Code Snippet:**

```cpp
// 1. Define the Operations
struct AddOp {
    __device__ float operator()(float a, float b) { return a + b; }
};

struct MaxOp {
    __device__ float operator()(float a, float b) { return a > b ? a : b; }
};

// 2. Define the Generic Kernel
template <typename Op>
__global__ void binary_op_kernel(float* a, float* b, float* out, Op op) {
    int idx = threadIdx.x;
    // The struct 'op' is called like a function!
    out[idx] = op(a[idx], b[idx]);
}

// 3. Launch
// Launch with Add
binary_op_kernel<<<1, 32>>>(d_a, d_b, d_out, AddOp());
// Launch with Max
binary_op_kernel<<<1, 32>>>(d_a, d_b, d_out, MaxOp());

```



# 3. The `CUDA_CHECK` Macro (The Debugger)

**The Concept:**
CUDA calls like `cudaMemcpy` return an error code (`cudaError_t`). If you ignore it, the program continues silently until it crashes later with a confusing message.
You cannot wrap every line in an `if` statement. You use a Macro.

**The Syntax:**
A `#define` that checks the return value. If it is not `cudaSuccess`, it prints the file, line number, and error message, then aborts.

**Code Snippet:**

```cpp
#include <iostream>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Usage:
// If this fails (e.g., OOM), it prints "CUDA Error at main.cpp:45 - Out of Memory" and quits.
CUDA_CHECK(cudaMalloc(&ptr, 1024));

```

# 4. Summary: The "Safe Launcher"

To pass Phase 4, combine these into a robust host function.

**Task:**
Write a `main` function that:

1. Uses `DeviceBuffer` to allocate memory (RAII).
2. Uses `CUDA_CHECK` for the memory copy.
3. Launches a kernel using a `SquareOp` functor.
4. Copies back and prints.

**Solution:**

```cpp
#include <iostream>

// --- Macro ---
#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("Err: %s\n", cudaGetErrorString(err)); exit(1); } }

// --- Functor ---
struct SquareOp {
    __device__ float operator()(float x) { return x * x; }
};

// --- Kernel ---
template <typename Func>
__global__ void apply_kernel(float* data, Func op) {
    int idx = threadIdx.x;
    data[idx] = op(data[idx]);
}

// --- RAII Wrapper ---
struct FloatBuffer {
    float* ptr;
    FloatBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float))); }
    ~FloatBuffer() { cudaFree(ptr); }
    operator float*() { return ptr; }
};

int main() {
    int N = 32;
    FloatBuffer d_buf(N); // Auto-allocates
    
    // Initialize host data
    float h_data[32];
    for(int i=0; i<32; i++) h_data[i] = (float)i;

    // Copy with check
    CUDA_CHECK(cudaMemcpy(d_buf, h_data, 32*sizeof(float), cudaMemcpyHostToDevice));

    // Launch with Functor
    apply_kernel<<<1, 32>>>(d_buf, SquareOp());
    
    // Check for Kernel Errors (Synchronous check)
    CUDA_CHECK(cudaGetLastError());

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_data, d_buf, 32*sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Result[5]: %f\n", h_data[5]); // Should be 25.0
    return 0; // d_buf destructor runs here, freeing VRAM.
}

```

**Checklist for success:**

* [ ] Did `FloatBuffer` clean up after itself?
* [ ] Did `SquareOp` work without changing the kernel code?
* [ ] Did `CUDA_CHECK` wrap the `cudaMemcpy`?
