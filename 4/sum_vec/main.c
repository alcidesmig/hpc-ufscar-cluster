#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#define TOL 0.0000001

#define ROOT 0
//#define DEBUG
#define N 1000000
#define N_REPEAT 10000

#define lli long long int

int main(int argc, char *argv[]) {
  int total, rank, err = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total);
  float *a, *b, *c, *a2, *b2, *c2;
  lli n = N, qt_per_thr = n / total;

  if (rank == ROOT) {
    printf("Size of workload: %d\n", total);

    a = (float *)malloc(sizeof(float) * n);
    b = (float *)malloc(sizeof(float) * n);
    c = (float *)malloc(sizeof(float) * n);
    float *res = (float *)malloc(sizeof(float) * n);
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
      a[i] = rand() % 10000;
      b[i] = rand() % 10000;
      res[i] = a[i] + b[i];
    }
    struct timeval time_start;
    struct timeval time_end;
    gettimeofday(&time_start, NULL);
    for (int r = 0; r < N_REPEAT; r++) {
      // MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
      qt_per_thr = n / total;
      a2 = (float *)malloc(sizeof(float) * qt_per_thr);
      b2 = (float *)malloc(sizeof(float) * qt_per_thr);
      c2 = (float *)malloc(sizeof(float) * qt_per_thr);
      MPI_Bcast(&qt_per_thr, 1, MPI_LONG_LONG_INT, ROOT, MPI_COMM_WORLD);
      MPI_Scatter(a, qt_per_thr, MPI_FLOAT, a2, qt_per_thr, MPI_FLOAT, 0,
                  MPI_COMM_WORLD);
      MPI_Scatter(b, qt_per_thr, MPI_FLOAT, b2, qt_per_thr, MPI_FLOAT, 0,
                  MPI_COMM_WORLD);
      // first block for master
      for (int i = 0; i < qt_per_thr; i++) {
        c2[i] = a2[i] + b2[i];
      }
      MPI_Gather(c2, qt_per_thr, MPI_FLOAT, c, qt_per_thr, MPI_FLOAT, ROOT,
                 MPI_COMM_WORLD);
      free(a2);
      free(b2);
      free(c2);
#ifdef DEBUG
      for (int i = 0; i < n; i++) {
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
      }
      printf("\n");
#endif
    }
    gettimeofday(&time_end, NULL);

    double exec_time =
        (double)(time_end.tv_sec - time_start.tv_sec) +
        (double)(time_end.tv_usec - time_start.tv_usec) / 1000000.0;
    printf("vectors added %d times in %lf seconds\n", N_REPEAT, exec_time);
  } else {
    for (int r = 0; r < N_REPEAT; r++) {
      MPI_Bcast(&qt_per_thr, 1, MPI_LONG_LONG_INT, ROOT, MPI_COMM_WORLD);
      a = (float *)malloc(sizeof(float) * qt_per_thr);
      b = (float *)malloc(sizeof(float) * qt_per_thr);
      c = (float *)malloc(sizeof(float) * qt_per_thr);
      MPI_Scatter(a, qt_per_thr, MPI_FLOAT, a, qt_per_thr, MPI_FLOAT, 0,
                  MPI_COMM_WORLD);
      MPI_Scatter(b, qt_per_thr, MPI_FLOAT, b, qt_per_thr, MPI_FLOAT, 0,
                  MPI_COMM_WORLD);
      for (int i = 0; i < qt_per_thr; i++) {
        c[i] = a[i] + b[i];
      }
      MPI_Gather(c, qt_per_thr, MPI_FLOAT, c, qt_per_thr, MPI_FLOAT, ROOT,
                 MPI_COMM_WORLD);
      free(a);
      free(b);
      free(c);
    }
  }
  MPI_Finalize();
  return 0;
}