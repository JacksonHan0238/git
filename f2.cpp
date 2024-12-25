#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>

#define N 10  
#define MAX_VAL 100  
void init_matrix(double A[N][N], double B[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % (MAX_VAL + 1); 
            B[i][j] = 0.0;  
        }
    }
}

void print_local_matrix(double local_B[][N+2], int rows_per_proc, int cols_per_proc, int rank) {
    printf("Processor %d local matrix B:\n", rank);
    for (int i = 1; i <= rows_per_proc; i++) {
        for (int j = 1; j <= cols_per_proc; j++) {
                printf("%.2f ", local_B[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double A[N][N], B[N][N];
    int i, j;
    int sqrt_p, rows_per_proc, cols_per_proc;
    double local_A[N+2][N+2], local_B[N+2][N+2];  

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sqrt_p = (int)sqrt(size);
    if (sqrt_p * sqrt_p != size) {
        if (rank == 0) {
            printf("Error: The number of processes must be a perfect square (1, 4, 9, 16, ...).\n");
        }
        MPI_Finalize();
        return -1;
    }

    double start_time = MPI_Wtime();

    rows_per_proc = N / sqrt_p;
    cols_per_proc = N / sqrt_p;

    if (rank == 0) {
        srand(time(NULL));  
        init_matrix(A, B);  
    }

    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int row_start = (rank / sqrt_p) * rows_per_proc;
    int col_start = (rank % sqrt_p) * cols_per_proc;

    for (i = 0; i < rows_per_proc + 2; i++) {
        for (j = 0; j < cols_per_proc + 2; j++) {
            if (i == 0 || i == rows_per_proc + 1 || j == 0 || j == cols_per_proc + 1) {
                local_A[i][j] = 0.0;
                local_B[i][j] = 0.0;
            } else {
                local_A[i][j] = A[row_start + i - 1][col_start + j - 1];
                local_B[i][j] = 0.0;
            }
        }
    }

    for (i = 1; i <= rows_per_proc; i++) {
        for (j = 1; j <= cols_per_proc; j++) {
            local_B[i][j] = (local_A[i - 1][j] + local_A[i][j + 1] + local_A[i + 1][j] + local_A[i][j - 1]) / 4.0;
        }
    }

    // 交换边界数据
    MPI_Status status;
    // 上邻居交换
    if (rank / sqrt_p > 0) { // 上邻居
        MPI_Sendrecv(&local_B[1][1], cols_per_proc, MPI_DOUBLE, rank - sqrt_p, 0,
                     &local_B[0][1], cols_per_proc, MPI_DOUBLE, rank - sqrt_p, 0, MPI_COMM_WORLD, &status);
    }
    // 下邻居交换
    if (rank / sqrt_p < sqrt_p - 1) { // 下邻居
        MPI_Sendrecv(&local_B[rows_per_proc][1], cols_per_proc, MPI_DOUBLE, rank + sqrt_p, 0,
                     &local_B[rows_per_proc + 1][1], cols_per_proc, MPI_DOUBLE, rank + sqrt_p, 0, MPI_COMM_WORLD, &status);
    }
    // 左邻居交换
    if (rank % sqrt_p > 0) { // 左邻居
        MPI_Sendrecv(&local_B[1][1], rows_per_proc, MPI_DOUBLE, rank - 1, 0,
                     &local_B[1][0], rows_per_proc, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
    }
    // 右邻居交换
    if (rank % sqrt_p < sqrt_p - 1) { // 右邻居
        MPI_Sendrecv(&local_B[1][cols_per_proc], rows_per_proc, MPI_DOUBLE, rank + 1, 0,
                     &local_B[1][cols_per_proc + 1], rows_per_proc, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
    }

   print_local_matrix(local_B, rows_per_proc, cols_per_proc, rank);

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Total parallel execution time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();

    return 0;
}
