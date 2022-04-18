#include <iostream>
#include <stdio.h>
#include <time.h>

#define NUM_ELEMENTS 4096

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(1);
  }
}

using namespace std;

// https://stackoverflow.com/questions/5689028/how-to-get-card-specs-programmatically-in-cuda
void systemInfo() {
  const int kb = 1024;
  const int mb = kb * kb;
  cout << "CUDA version:   v" << CUDART_VERSION << endl;

  int devCount;
  cudaGetDeviceCount(&devCount);
  cout << "CUDA Devices: " << endl << endl;

  for (int i = 0; i < devCount; ++i) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);
    cout << i << ": " << props.name << ": " << props.major << "." << props.minor
         << endl;
    cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
    cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb"
         << endl;
    cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
    cout << "  Block registers: " << props.regsPerBlock << endl << endl;

    cout << "  Warp size:         " << props.warpSize << endl;
    cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
    cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", "
         << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]"
         << endl;
    cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", "
         << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]"
         << endl;
    cout << endl;
  }
}

__host__ __device__ inline void setAt(int *m, int i, int j, int v) {
  *(m + i * NUM_ELEMENTS + j) = v;
}

__host__ __device__ inline int getAt(int *m, int i, int j) {
  return *(m + i * NUM_ELEMENTS + j);
}

__global__ void warshall_gpu(int *m, int num_threads) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ int m_column_shared[NUM_ELEMENTS];
  int common_m_i_k = getAt(m, i, k) == 1;

  if (k >= NUM_ELEMENTS || i >= NUM_ELEMENTS)
    return;
  for (int j = threadIdx.x * (NUM_ELEMENTS / num_threads);
       j < (threadIdx.x + 1) * (NUM_ELEMENTS / num_threads) && j < NUM_ELEMENTS;
       j++) {
    m_column_shared[j] = getAt(m, k, j);
  }
  __syncthreads();

  if(!common_m_i_k) return;


  for (int j = 0; j < NUM_ELEMENTS; j++) {
    if (m_column_shared[j] == 1) {
      setAt(m, i, j, 1);
    }
  }
}

void initMatrix(int *data, unsigned size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      data[i * size + j] = ((int)rand()) % 2;
    }
  }
}

int *warshallGPU(int *matrix, int thread_limit_per_block_sqrt) {
  size_t _alloc = NUM_ELEMENTS * NUM_ELEMENTS * sizeof(int);
  int *d_matrix, *output = (int *)malloc(_alloc);
  gpuErrchk(cudaMalloc(&d_matrix, _alloc));
  gpuErrchk(cudaMemcpy(d_matrix, matrix, _alloc, cudaMemcpyHostToDevice));

  int blockSize = (NUM_ELEMENTS / thread_limit_per_block_sqrt) + 1;
  dim3 threadsPerBlock(thread_limit_per_block_sqrt,
                       thread_limit_per_block_sqrt);
  dim3 numBlocks(blockSize, blockSize, 1);

  warshall_gpu<<<numBlocks, threadsPerBlock>>>(d_matrix,
                                               thread_limit_per_block_sqrt);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(output, d_matrix, _alloc, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_matrix));

  return output;
}

void mainWarshall() {

  int *d = (int *)malloc(NUM_ELEMENTS * NUM_ELEMENTS * sizeof(int));
  initMatrix(d, NUM_ELEMENTS);
  for (int thread_limit_per_block_sqrt = 32; thread_limit_per_block_sqrt >= 1;
       thread_limit_per_block_sqrt >>= 1) {
    printf("Threads %dx%d\n", thread_limit_per_block_sqrt,
           thread_limit_per_block_sqrt);
    clock_t start = clock();
    int *output = warshallGPU(d, thread_limit_per_block_sqrt);
    clock_t stop = clock();
    free(output);
    double t = (stop - start) / (float)CLOCKS_PER_SEC;
    printf("GPU time: %f s\n", t);
  }
  free(d);
}

int main(void) {
  systemInfo();
  mainWarshall();
  return 0;
}
