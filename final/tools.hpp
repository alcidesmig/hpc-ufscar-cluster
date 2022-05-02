#ifndef TOOLS_H
#define TOOLS_H

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

namespace tools {
void systemInfo();
void save_grid(int rows, int cols, float *matrix);
void initMatrix(float **prev_base, float **next_base, int rows, int cols,
                float wave_velocity);
void swapf(float **x, float **y);
} // namespace tools
#endif