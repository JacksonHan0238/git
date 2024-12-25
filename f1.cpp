#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>

#define MATRIX_SIZE 100  
#define BLOCK_DIM(row, size) ((row) / (size))  

void calculate(const double* input_matrix, double* output_matrix, int rows, int cols) {
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            output_matrix[i * cols + j] = (input_matrix[(i - 1) * cols + j] +
                                           input_matrix[(i + 1) * cols + j] +
                                           input_matrix[i * cols + (j - 1)] +
                                           input_matrix[i * cols + (j + 1)]) / 4.0;
        }
    }
}

void display_matrix(const double* local_matrix, int local_rows, int local_cols, int rank, int dims[2], MPI_Comm grid_comm) {
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords); 

    printf("Rank %d (row %d, col %d):\n", rank, coords[0], coords[1]);
    for (int i = 1; i < local_rows - 1; i++) {
        for (int j = 1; j < local_cols - 1; j++) {
            printf("%6.2f ", local_matrix[i * local_cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = { 0, 0 };
    int periods[2] = { false, false }; 
    MPI_Dims_create(size, 2, dims);
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    int local_rows = BLOCK_DIM(MATRIX_SIZE, dims[0]) + 2;  
    int local_cols = BLOCK_DIM(MATRIX_SIZE, dims[1]) + 2;  

    srand(time(NULL) + rank);
    double* matrix_in = new double[local_rows * local_cols];
    double* matrix_out = new double[local_rows * local_cols];
    for (int i = 0; i < local_rows * local_cols; i++) {
        matrix_in[i] = rand() % 100;
        matrix_out[i] = 0;
    }

    int neighbors[4];
    MPI_Cart_shift(grid_comm, 0, 1, &neighbors[0], &neighbors[1]);  
    MPI_Cart_shift(grid_comm, 1, 1, &neighbors[2], &neighbors[3]);  

    MPI_Request reqs[8];
    for (int i = 0; i < 8; i++) reqs[i] = MPI_REQUEST_NULL;

    if (neighbors[0] >= 0) {
        MPI_Isend(&matrix_in[1 * local_cols], local_cols, MPI_DOUBLE, neighbors[0], 0, grid_comm, &reqs[0]);
        MPI_Irecv(&matrix_in[0 * local_cols], local_cols, MPI_DOUBLE, neighbors[0], 0, grid_comm, &reqs[1]);
    }
    if (neighbors[1] >= 0) {
        MPI_Isend(&matrix_in[(local_rows - 2) * local_cols], local_cols, MPI_DOUBLE, neighbors[1], 0, grid_comm, &reqs[2]);
        MPI_Irecv(&matrix_in[(local_rows - 1) * local_cols], local_cols, MPI_DOUBLE, neighbors[1], 0, grid_comm, &reqs[3]);
    }

        if (neighbors[2] >= 0) {
        for (int i = 1; i < local_rows - 1; i++) {
            MPI_Isend(&matrix_in[i * local_cols + 1], 1, MPI_DOUBLE, neighbors[2], 0, grid_comm, &reqs[4]);
            MPI_Irecv(&matrix_in[i * local_cols + 0], 1, MPI_DOUBLE, neighbors[2], 0, grid_comm, &reqs[5]);
        }
    }
    if (neighbors[3] >= 0) {
        for (int i = 1; i < local_rows - 1; i++) {
            MPI_Isend(&matrix_in[i * local_cols + (local_cols - 2)], 1, MPI_DOUBLE, neighbors[3], 0, grid_comm, &reqs[6]);
            MPI_Irecv(&matrix_in[i * local_cols + (local_cols - 1)], 1, MPI_DOUBLE, neighbors[3], 0, grid_comm, &reqs[7]);
        }
    }

    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

    double start_time = MPI_Wtime();

    calculate(matrix_in, matrix_out, local_rows, local_cols);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total computation time: %f seconds\n", elapsed_time);
    }

    display_matrix(matrix_out, local_rows, local_cols, rank, dims, grid_comm);

    delete[] matrix_in;
    delete[] matrix_out;

    MPI_Finalize();
    return 0;
}
