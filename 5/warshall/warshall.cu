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

double assertSum(int *data, unsigned n) {
  double soma = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      soma += data[i * n + j];
    }
  }
  return soma;
}

void initMatrix(int *data, unsigned size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      //   data[i * size + j] = (int)( rand() & 0xFF )/10.0f;
      data[i * size + j] = ((int)rand()) % 2;
    }
  }
}

void _warshallCPU(int *m, unsigned n) {
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (m[k * n + j] == 1 && m[i * n + k] == 1)
          m[i * n + j] = 1;
      }
    }
  }
}

double warshallCPU(int *A, unsigned n) {
  int *F = (int *)malloc(sizeof(int) * n * n);
  memcpy(F, A, sizeof(int) * n * n);
  double t;
  clock_t start = clock();
  _warshallCPU(F, n);
  clock_t stop = clock();
  t = (stop - start) / (float)CLOCKS_PER_SEC;
  printf("CPU time: %f s\n\n", t);
  double s_ = assertSum(F, n);
  free(F);
  return s_;
}

__global__ void warshall_gpu(int *m) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (k >= NUM_ELEMENTS || i >= NUM_ELEMENTS)
    return;
  for (int j = 0; j < NUM_ELEMENTS; j++) {
    if (getAt(m, k, j) == 1 && getAt(m, i, k) == 1) {
      setAt(m, i, j, 1);
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

  warshall_gpu<<<numBlocks, threadsPerBlock>>>(d_matrix);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(output, d_matrix, _alloc, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_matrix));
  
  return output;
}

void mainWarshall() {

  int *d = (int *)malloc(NUM_ELEMENTS * NUM_ELEMENTS * sizeof(int));
  initMatrix(d, NUM_ELEMENTS);
  double cpu = warshallCPU(d, NUM_ELEMENTS);
  for (int thread_limit_per_block_sqrt = 32; thread_limit_per_block_sqrt >= 1;
       thread_limit_per_block_sqrt >>= 1) {
    printf("Threads %dx%d\n", thread_limit_per_block_sqrt,
           thread_limit_per_block_sqrt);
    clock_t start = clock();
    int *output = warshallGPU(d, thread_limit_per_block_sqrt);
    clock_t stop = clock();
    double gpu = assertSum(output, NUM_ELEMENTS);
    free(output);
    double t = (stop - start) / (float)CLOCKS_PER_SEC;
    printf("GPU time: %f s\n", t);
    printf("valid result %lf(cpu)==%lf(gpu) -> %d\n\n", cpu, gpu, cpu == gpu);
  }
  free(d);
}

int main(void) {
  systemInfo();
  mainWarshall();
  return 0;
}
