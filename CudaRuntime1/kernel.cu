
#include "cuda_utils.h"

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
        TIMEIT(
            for (int i = 0; i < a.size; i++) {
                c.ptr_host[i] = a.ptr_host[i] + b.ptr_host[i];
            }
        );
    }
}



int addWithCuda2() {
    // Choose which GPU to run on, change this on a multi-GPU system.
    ASSERT(cudaSetDevice(0) == cudaSuccess);

    tensor2D<int> c(256, 4096);
    tensor2D<int> a(256, 4096);
    tensor2D<int> b(256, 4096);
    for (int i = 0; i < a.size; i++) {
        a.ptr_host[i] = i;
        b.ptr_host[i] = i;
    }
    
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




cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void transpose(int *dst, int* src, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    int src_index = x + y * W;
    int dst_index = y + x * H;
    dst[dst_index] = src[src_index];
}

// this HOST side wrapper can have same name as DEVICE side kernel
// Threads per Block has 1024 limitations, so each block can only transpose [32,32] sub-matrix
// Host side wrapper needs to decompose the original problem into [32,32] sub problems.
// (OR let each thread do multiple sub-problems within a single call, but this is not friendly to
// exploit full capability of HW since target GPU may have multiple SMs capable of excuting multiple sub-problems in parallel)
// 
// But host side should decompse the problem once for all and describe the decomposed problems in gridDim/blockDim and call
// kernel only once, and CUDA framework will handle the schedulling for us.
// 
// src: [H, W]
void _transpose(int* dst, int* src, int H, int W) {
    dim3 blockDim(32, 32);
    dim3 gridDim((W + 31) / 32, (H + 31) / 32);
    transpose << <gridDim, blockDim >> > (dst, src, H, W);
}



void transpose_acc_test(int H, int W) {
    // test correctness

}

const int arraySize = 16*1024*1024;
const int a[arraySize] = { 1, 2, 3, 4, 5 };
const int b[arraySize] = { 10, 20, 30, 40, 50 };
int c[arraySize] = { 0 };

int main()
{
    return addWithCuda2();
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    int* dev_d = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_d, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.

    for (int i = 0; i < 10; i++) {
        // flattened 1D
        dim3 threadsPerBlock(256);
        dim3 numBlocks(size / threadsPerBlock.x);
        addKernel << <numBlocks, threadsPerBlock >> > (dev_c, dev_a, dev_b);
        assert(cudaGetLastError() == cudaSuccess);


        // problem is : on my 1070,
        // 2D shape (size_h, size_w)
        _transpose(dev_d, dev_c, 1024, size/1024);

        assert(cudaGetLastError() == cudaSuccess);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_d, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
