/*
    This program solves Laplace's equation on a regular 2D grid using simple Jacobi iteration.

    The stencil calculation stops when  iter > ITER_MAX
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define ITER_MAX 3000          // number of maximum iterations
#define CONV_THRESHOLD 1.0e-5f // threshold of convergence
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// matrix to be solved
double **grid;

// auxiliary matrix
double **new_grid;

// size of each side of the grid
int size;

// number of threads
int num_threads;

// return the maximum value
double max(double a, double b)
{
    if (a > b)
        return a;
    return b;
}

// return the absolute value of a number
double absolute(double num)
{
    if (num < 0)
        return -1.0 * num;
    return num;
}

// allocate memory for the grid
void allocate_memory()
{
    grid = (double **)malloc(size * sizeof(double *));
    new_grid = (double **)malloc(size * sizeof(double *));

    for (int i = 0; i < size; i++)
    {
        grid[i] = (double *)malloc(size * sizeof(double));
        new_grid[i] = (double *)malloc(size * sizeof(double));
    }
}

// initialize the grid
void initialize_grid()
{
    // seed for random generator
    srand(10);

    int linf = size / 2;
    int lsup = linf + size / 10;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            // inicializa região de calor no centro do grid
            if (i >= linf && i < lsup && j >= linf && j < lsup)
                grid[i][j] = 100;
            else
                grid[i][j] = 0;
            new_grid[i][j] = 0.0;
        }
    }
}

// save the grid in a file
void save_grid()
{

    char file_name[30];
    sprintf(file_name, "grid_laplace.txt");

    // save the result
    FILE *file;
    file = fopen(file_name, "w");

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            fprintf(file, "%lf ", grid[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

typedef struct
{
    int iInit;
    int iEnd;
} Param;

void *calculate(void *_p)
{
    Param *p = (Param *)_p;
    for (int i = p->iInit; i < p->iEnd; i++)
    {
        for (int j = 1; j < size - 1; j++)
        {
            new_grid[i][j] = 0.25 * (grid[i][j + 1] + grid[i][j - 1] +
                                     grid[i - 1][j] + grid[i + 1][j]);
        }
        // printf("i: %d\n", i);
    }
}

void *post_process(void *_p)
{
    Param *p = (Param *)_p;
    for (int i = p->iInit; i < p->iEnd; i++)
    {
        for (int j = 1; j < size - 1; j++)
        {
            grid[i][j] = new_grid[i][j];
        }
    }
}

int main(int argc, char *argv[])
{

    if (argc != 3)
    {
        printf("Usage: ./laplace_par N T\n");
        printf("N: The size of each side of the domain (grid)\n");
        printf("T: Number of threads\n");
        exit(-1);
    }

    // variables to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    size = atoi(argv[1]);
    num_threads = atoi(argv[2]);

    // allocate memory to the grid (matrix)
    allocate_memory();

    // set grid initial conditions
    initialize_grid();

    double err = 1.0;
    int iter = 0;

    // get the start time
    gettimeofday(&time_start, NULL);

    // Jacobi iteration
    // This loop will end if either the maximum change reaches below a set threshold (convergence)
    // or a fixed number of maximum iterations have completed
    while (iter <= ITER_MAX)
    {

        // calculates the Laplace equation to determine each cell's next value
        // kernel 1
        pthread_t t_id[num_threads];

        // number of lines to be processed per thread
        int num_lines_per_thread = (size - 1) / num_threads;
        int tid_counter = 0;
        // process
        for (int i = 1; i < size - 1; i += num_lines_per_thread)
        {
            Param *p = (Param *)malloc(sizeof(Param));
            p->iInit = i;
            p->iEnd = MIN(i + num_lines_per_thread - 1, size-1);
            pthread_create(&t_id[tid_counter++], NULL, calculate, p);
        }

        for (int i = 0; i < num_threads; i++)
        {
            pthread_join(t_id[i], NULL);
        }

        // copy the next values into the working array for the next iteration
        // kernel 2
        tid_counter = 0;
        pthread_t t2_id[num_threads];
        for (int i = 1; i < size - 1; i += num_lines_per_thread)
        {
            Param *p = (Param *)malloc(sizeof(Param));
            p->iInit = i;
            p->iEnd = MIN(i + num_lines_per_thread - 1, size-1);
            pthread_create(&t2_id[tid_counter++], NULL, post_process, p);
        }

        for (int i = 0; i < num_threads; i++)
        {
            pthread_join(t2_id[i], NULL);
        }

        iter++;
    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double)(time_end.tv_sec - time_start.tv_sec) +
                       (double)(time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    //save the final grid in file
    save_grid();

    printf("\nKernel executed in %lf seconds with %d iterations \n", exec_time, iter);

    return 0;
}