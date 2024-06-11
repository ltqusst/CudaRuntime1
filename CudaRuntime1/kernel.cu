
#include "cuda_utils.h"
#include <immintrin.h>
/*
* CUDA and Application to Task-Based Programming (part 1) | Eurographics'2021 Tutorial
https://cuda-tutorial.github.io/
part1 https://www.youtube.com/watch?v=6kT7vVHCZIc
part2 https://www.youtube.com/watch?v=mrDWmnXC5Ck

example of SMs/CUDA-Cores in GP104’s Architecture: 
https://www.anandtech.com/show/10325/the-nvidia-geforce-gtx-1080-and-1070-founders-edition-review/4
https://www.anandtech.com/show/11172/nvidia-unveils-geforce-gtx-1080-ti-next-week-699

 - CUDA core is just a scalar ALU;

 - GPU executes multiple threads in time-division multiplexing fasion, but unlike CPU:
      - context-switching of HW threads is designed to be very fast (on 1 cycle level);
      - HW-threads has all states in register files, never goto memory;
      - HW-scheduler is designed to switching context (unlike OS running on CPU using SW-scheduler);
      - number of threads supported are limited by register file size & HW-scheduler capability;
      - HW-threads are grouped in unit of 32 into Warp, to get higher ALU throughput.
      - number of wraps/HW-threads supported is far more than number of CUDA cores(to hide mem-latency).
 
 for example on my GTX-1070, 2048 threads (or 64 warps) per SM are supported,
 but each SM has only 4 Warp schedulers, 16x oversubscride.


 GPU Memory hierarchy
 GPU cache is designed with significant different purpose from CPU's
 https://www.rastergrid.com/blog/gpu-tech/2021/01/understanding-gpu-caches/

Cache Coherency:
    As a result, GPU caches are usually incoherent, and require explicit flushing
    and/or invalidation of the caches in order to recohere (i.e. to have a coherent
    view of the data between the GPU cores and/or other devices).
     - shader invocations within individual draw or compute dispatch commands
       that run on different GPU cores may see incoherent views of the same memory data,
       unless using coherent resources (see later), or issuing memory barriers within
       the shader that flush/invalidate the per core caches
     - subsequent draw or compute dispatch commands may see incoherent views of the
       same memory data, unless appropriate memory barriers are issued through the API
       that flush/invalidate the appropriate caches
     - GPU commands and operations performed by other devices (e.g. CPU reads/writes)
       may see incoherent views of the same memory data, unless appropriate synchronization
       primitives (e.g. fences or semaphores) are used to synchronize them, which implicitly
       insert the necessary memory barriers

Per Core Instruction Cache:
    One thing to keep in mind from performance point of view is that on GPUs
    an instruction cache miss can potentially stall thousands of threads
    instead of just one, as in the case of CPUs, so generally it’s highly recommended
    for shader/kernel code to be small enough to completely fit into this cache.

Per Core Data Cache:
    "Thus reuse of cached data on GPUs usually does not happen in the time domain
    as in case of CPUs (i.e. subsequent instructions of the same thread accessing
    the same data multiple times), but rather in the spatial domain (i.e. instructions
    of different threads on the same core accessing the same data).

    Accordingly, using memory as temporary storage and relying on caching for fast
    repeated access is not a good idea on GPUs, unlike on CPUs. However, the larger
    register space and the availability of shared memory make up for that in practice."

*/

__global__ void TensorAddKernel(int* c, int* a, int* b, int d0, int d1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

// OP dispatcher
template<typename T>
void tensor_add(tensor2D<T>& c, tensor2D<T>& a, tensor2D<T>& b) {
    ASSERT(c.shape[0] == a.shape[0] && c.shape[1] == a.shape[1]);
    ASSERT(c.shape[0] == b.shape[0] && c.shape[1] == b.shape[1]);
    ASSERT(a.on_device == b.on_device);

    // x is inner-most dimension
    if (a.on_device) {
        c.to_dev(false);
        dim3 blockDim(32, 32);
        dim3 gridDim((c.shape[1] + 31) / 32, (c.shape[0] + 31) / 32);
        TIMEIT(
            TensorAddKernel << <gridDim, blockDim >> > (c.ptr_dev, a.ptr_dev, b.ptr_dev, a.shape[0], a.shape[1]);
        );
    }
    if (!a.on_device) {
        // CPU side
        c.to_host(false);
        TIMEIT_BEGIN();
        #pragma omp parallel for
        for (int i = 0; i < a.size; i+=8) {
            __m256i ra = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a.ptr_host + i));
            __m256i rb = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b.ptr_host + i));
            __m256i rc = _mm256_add_epi32(ra, rb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c.ptr_host + i), rc);
        }
        TIMEIT_END();
    }
}



int addWithCuda2() {

    tensor2D<int> c(256, 4096);
    tensor2D<int> a(256, 4096);
    tensor2D<int> b(256, 4096);
    for (int i = 0; i < a.size; i++) {
        a.ptr_host[i] = i;
        b.ptr_host[i] = i;
    }
    
    tensor_add(c, a, b);
    tensor_add(c, a, b);
    tensor_add(c, a, b);
    tensor_add(c, a, b);
    std::cout << "c=" << c << std::endl;
    a.to_dev();
    b.to_dev();
    tensor_add(c, a, b);
    tensor_add(c, c, b);
    tensor_add(c, c, b);
    tensor_add(c, c, b);
    tensor_add(c, c, b);
    tensor_add(c, c, b);
    c.to_host();
    std::cout << "c=" << c << std::endl;
    return 0;
}

// https://www.youtube.com/watch?v=6kT7vVHCZIc
__device__ float result = 0.1f;
__global__ void reduceAtomicGlobal(const float* input, int N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N)
        atomicAdd(&result, input[id]);
}

// this version actually is slower than reduceAtomicGlobal
// since in each thread it brings too much overheads before calling `atomicAdd`
// it only faster if partial sum is used more than once
__global__ void reduceAtomicShared(const float* input, int N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory is allocated per thread block, so all threads in the block
    // have access to the same shared memory.
    __shared__ float partial_sum;
    // thus only the first thread clears it
    if (threadIdx.x == 0) partial_sum = 0;
    __syncthreads();

    if (id < N) atomicAdd(&partial_sum, input[id]);
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(&result, partial_sum);
}

#define BLOCKSIZE 1024
__global__ void reduceParallelShared(const float* input, int N) {
    __shared__ float data[BLOCKSIZE];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // load input into shared data for current thread-block
    data[threadIdx.x] = (id < N ? input[id] : 0);
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        __syncthreads();
        if (threadIdx.x < s)
            data[threadIdx.x] += data[threadIdx.x + s];
    }
    if (threadIdx.x == 0)
        atomicAdd(&result, data[0]);
}

__global__ void reduceParallelSharedShfl(const float* input, int N) {
    __shared__ float data[BLOCKSIZE];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // load input into shared data for current thread-block
    data[threadIdx.x] = (id < N ? input[id] : 0);
    for (int s = blockDim.x / 2; s > 16; s /= 2) {
        __syncthreads();
        // SIMD horizontal add
        if (threadIdx.x < s)
            data[threadIdx.x] += data[threadIdx.x + s];
    }
    float x = data[threadIdx.x];
    if (threadIdx.x < 32) {
        // SIMD horizontal reduce
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
        x += __shfl_sync(0xFFFFFFFF, x, 1);
    }
    if (threadIdx.x == 0)
        atomicAdd(&result, x);
}

template<typename T>
T tensor_reduce(tensor2D<T>& c) {
    if (c.on_device) {
        dim3 gridSize((c.size + 1023) / 1024);
        dim3 blockSize(1024);
        float Y = 0.0f;
        uint64_t bytes = c.size * sizeof(T);
        uint64_t flops = c.size;

        CUDA_CALL(cudaMemcpyToSymbol(result, &Y, sizeof(Y)));
        TIMEIT_BEGIN("reduceAtomicGlobal", bytes, flops);
        reduceAtomicGlobal << <gridSize, blockSize>> > (c.ptr_dev, c.size); // 3ms
        TIMEIT_END();

        CUDA_CALL(cudaMemcpyToSymbol(result, &Y, sizeof(Y)));
        TIMEIT_BEGIN("reduceAtomicShared", bytes, flops);
        reduceAtomicShared << <gridSize, blockSize >> > (c.ptr_dev, c.size); // 20ms
        TIMEIT_END();

        CUDA_CALL(cudaMemcpyToSymbol(result, &Y, sizeof(Y)));
        TIMEIT_BEGIN("reduceParallelShared", bytes, flops);
        reduceParallelShared << <gridSize, blockSize >> > (c.ptr_dev, c.size); // 100us
        TIMEIT_END();

        CUDA_CALL(cudaMemcpyToSymbol(result, &Y, sizeof(Y)));
        TIMEIT_BEGIN("reduceParallelSharedShfl", bytes, flops);
        reduceParallelSharedShfl << <gridSize, blockSize >> > (c.ptr_dev, c.size); // 64us
        TIMEIT_END();

        CUDA_CALL(cudaMemcpyFromSymbol(&Y, result, sizeof(result)));
        return Y;
    }
    else {
        TIMEIT_BEGIN("tensor_reduce_CPU");
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < c.size; i++)
            sum += c.ptr_host[i];
        TIMEIT_END();
        return sum;
    }
}



void testReduce() {
    tensor2D<float> c(1024, 1024);
    for (int i = 0; i < c.size; i++)
        c.ptr_host[i] = (i % 16) - 8;
    auto s0 = tensor_reduce(c);
    c.to_dev();
    auto s1 = tensor_reduce(c);
    printf("s0(host)=%f, s1(device)=%f\n", s0, s1);

    c.to_host();
    tensor_reduce(c);
    tensor_reduce(c);
    tensor_reduce(c);
    c.to_dev();
    tensor_reduce(c);
    tensor_reduce(c);
    tensor_reduce(c);
}


__global__ void testCpy(int* in, int* out, int N) {
    int block_offset = blockIdx.x * blockDim.x;
    int warp_offset = 32 * (threadIdx.x / 32);
    int lane_id = threadIdx.x % 32;
    int id = (block_offset + warp_offset + lane_id) % N;
    out[id] = in[id];
}

void test_cpy() {
    tensor2D<int> c(10240, 1024);
    tensor2D<int> d(10240, 1024);
    for (int i = 0; i < c.size; i++)
        c.ptr_host[i] = (i % 16) - 8;
    c.to_dev();
    d.to_dev(false);
    dim3 gridSize((c.size + 1023) / 1024);
    dim3 blockSize(1024);
    uint64_t bytes = c.size * sizeof(int);
    for (int i = 0; i < 5; i++) {
        TIMEIT_BEGIN("testCpy", bytes);
        testCpy << <gridSize, blockSize >> > (c.ptr_dev, d.ptr_dev, c.size);
        TIMEIT_END();
    }
}

int main()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    ASSERT(cudaSetDevice(0) == cudaSuccess);

    //test_cpy();
    testReduce();
    //addWithCuda2();

    TIMEIT_FINISH();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    ASSERT(cudaDeviceReset() == cudaSuccess);

    return 0;
}
