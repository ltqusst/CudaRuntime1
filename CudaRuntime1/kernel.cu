
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

int main()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    ASSERT(cudaSetDevice(0) == cudaSuccess);

    addWithCuda2();


    TIMEIT_FINISH();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    ASSERT(cudaDeviceReset() == cudaSuccess);

    return 0;
}
