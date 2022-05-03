#include <stdio.h>
#include <time.h>

#include "tools.hpp"
//#define DEBUG
using namespace std;

__global__ void _acoustic_wave(float *prev_base, float *next_base,
                               int iterations, int rows, int cols,
                               int num_threads, int stencil_radius,
                               float dxSquared, float dySquared,
                               float dtSquared, float wave_velocity_pow2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= rows - stencil_radius || j <= stencil_radius ||
      i <= stencil_radius || j >= cols - stencil_radius)
    return;

  int current = i * cols + j;

  int prev_base_current = 2.0 * prev_base[current];

  // neighbors in the horizontal direction +
  // neighbors in the vertical direction
  // * dtSquared * wave_velocity * wave_velocity
  next_base[current] =
      ((((prev_base[current + 1] - prev_base_current + prev_base[current - 1]) /
         dxSquared) +
        ((prev_base[current + cols] - prev_base_current +
          prev_base[current - cols]) /
         dySquared)) *
       (dtSquared * wave_velocity_pow2)) +
      ((prev_base_current)-next_base[current]);
}

float *acoustic_wave(float *in_prev_base, float *in_next_base, int rows,
                     int cols, int thread_limit_per_block_sqrt,
                     int timesteps_number, float delta_t, float delta_x,
                     float delta_y, float wave_velocity, int stencil_radius) {
#ifdef DEBUG
  printf("Starting accoustic_wave...\n");
#endif
  size_t _alloc_size = rows * cols * sizeof(float);
  float d_t_sqr = delta_t * delta_t, d_x_sqr = delta_x * delta_x,
        d_y_sqr = delta_y * delta_y, *prev_base, *next_base,
        *output = (float *)malloc(_alloc_size);
  gpuErrchk(cudaMalloc(&prev_base, _alloc_size));
#ifdef DEBUG
  printf("prev_base memory allocated: GPU\n");
#endif
  gpuErrchk(
      cudaMemcpy(prev_base, in_prev_base, _alloc_size, cudaMemcpyHostToDevice));
#ifdef DEBUG
  printf("prev_base memory set: GPU\n");
#endif
  gpuErrchk(cudaMalloc(&next_base, _alloc_size));
#ifdef DEBUG
  printf("next_base memory allocated: GPU\n");
#endif
  gpuErrchk(
      cudaMemcpy(next_base, in_next_base, _alloc_size, cudaMemcpyHostToDevice));
#ifdef DEBUG
  printf("next_base memory set: GPU\n");
#endif

#ifdef DEBUG
  printf("All memory allocated: GPU\n");
#endif
  int blockSizeX = (rows / thread_limit_per_block_sqrt) + 1,
      blockSizeY = (cols / thread_limit_per_block_sqrt) + 1;
  dim3 threadsPerBlock(thread_limit_per_block_sqrt,
                       thread_limit_per_block_sqrt),
      numBlocks(blockSizeX, blockSizeY, 1);
#ifdef DEBUG
  printf("blockSizeX %d, blockSizeY %d, thread_limit_per_block_sqrt %d: GPU\n",
         blockSizeX, blockSizeY, thread_limit_per_block_sqrt);
#endif

  int iterations = (int)((timesteps_number / 1000.0) / delta_t);
#ifdef DEBUG
  printf("Number of iterations: %d\n", iterations);
#endif
  for (int i = 0; i < iterations; i++) {
    _acoustic_wave<<<numBlocks, threadsPerBlock>>>(
        prev_base, next_base, iterations, rows, cols,
        thread_limit_per_block_sqrt, stencil_radius, d_x_sqr, d_y_sqr, d_t_sqr,
        wave_velocity*wave_velocity);
    tools::swapf(&next_base, &prev_base);
  }
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(output, next_base, _alloc_size, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(prev_base));
  gpuErrchk(cudaFree(next_base));

#ifdef DEBUG
  float sum = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      sum += output[i * cols + j];
    }
  }
  printf("output sum: %f\n", sum);
#endif

  return output;
}

void applyWave(char *argv[]) {

  // number of rows of the grid
  int rows = atoi(argv[1]);
  // number of columns of the grid
  int cols = atoi(argv[2]);
  // number of timesteps
  int timesteps_number = atoi(argv[3]);
  float delta_t = 0.0070710676f;
  float delta_x = 15.0f;
  float delta_y = 15.0f;
  float wave_velocity = 1500.0f;
  int stencil_radius = 1;
#ifdef DEBUG
  printf("Starting initMatrix\n");
#endif
  float *prev_base, *next_base;
  tools::initMatrix(&prev_base, &next_base, rows, cols, wave_velocity);
  for (int thread_limit_per_block_sqrt = 32; thread_limit_per_block_sqrt >= 4;
       thread_limit_per_block_sqrt >>= 1) {
    printf("Threads %dx%d\n", thread_limit_per_block_sqrt,
           thread_limit_per_block_sqrt);
    clock_t start = clock();
    float *output =
        acoustic_wave(prev_base, next_base, rows, cols,
                      thread_limit_per_block_sqrt, timesteps_number, delta_t,
                      delta_x, delta_y, wave_velocity, stencil_radius);
    clock_t stop = clock();
    tools::save_grid(rows, cols, output);
    free(output);
    double t = (stop - start) / (float)CLOCKS_PER_SEC;
    printf("GPU time: %f s for %d threads\n", t,
           thread_limit_per_block_sqrt * thread_limit_per_block_sqrt);
  }
  free(prev_base);
  free(next_base);
}

int main(int argc, char *argv[]) {
  tools::systemInfo();
  applyWave(argv);
  return 0;
}