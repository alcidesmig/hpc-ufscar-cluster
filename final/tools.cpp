#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;

namespace tools {
// https://stackoverflow.com/questions/5689028/how-to-get-card-specs-programmatically-in-cuda
void systemInfo() {
  const int kb = 1024;
  const int mb = kb * kb;

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
void save_grid(int rows, int cols, float *matrix) {

  system("mkdir -p wavefield");

  char file_name[64];
  sprintf(file_name, "wavefield/wavefield.txt");

  // save the result
  FILE *file;
  file = fopen(file_name, "w");

  for (int i = 0; i < rows; i++) {

    int offset = i * cols;

    for (int j = 0; j < cols; j++) {
      fprintf(file, "%f ", matrix[offset + j]);
    }
    fprintf(file, "\n");
  }

  fclose(file);

  system("python3 plot.py");
}

void initMatrix(float **prev_base, float **next_base, int rows, int cols,
                float wave_velocity) {
  // represent the matrix of wavefield as an array
  *prev_base = (float *)malloc(rows * cols * sizeof(float));
  *next_base = (float *)malloc(rows * cols * sizeof(float));
#ifdef DEBUG
  printf("prev_base and next_base allocated\n");
#endif
  // define source wavelet
  float wavelet[12] = {0.016387336, -0.041464937, -0.067372555, 0.386110067,
                       0.812723635, 0.416998396,  0.076488599,  -0.059434419,
                       0.023680172, 0.005611435,  0.001823209,  -0.000720549};

  // initialize matrix
  for (int i = 0; i < rows; i++) {
    int offset = i * cols;
    for (int j = 0; j < cols; j++) {
      (*prev_base)[offset + j] = 0.0f;
      (*next_base)[offset + j] = 0.0f;
    }
  }

  // add a source to initial wavefield as an initial condition
  for (int s = 11; s >= 0; s--) {
    for (int i = rows / 2 - s; i < rows / 2 + s; i++) {
      int offset = i * cols;
      for (int j = cols / 2 - s; j < cols / 2 + s; j++)
        (*prev_base)[offset + j] = wavelet[s];
    }
  }
#ifdef DEBUG
  float sum = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int offset = i * cols + j;
      sum += (*prev_base)[offset];
      if ((*prev_base)[offset] != 0.0)
        printf("%f ", (*prev_base)[offset]);
    }
  }
  printf("\nprev_base sum: %f\n", sum);
#endif
}
void swapf(float **x, float **y) {
  float *tmp = *x;
  *x = *y;
  *y = tmp;
}
} // namespace tools