#include <iostream>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>

#define N 256  

void displayMatrix(const std::vector<int>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void getSubmatrix(const std::vector<int>& full_matrix, std::vector<int>& sub_matrix, int row, int col, int block_size) {
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            sub_matrix[i * block_size + j] = full_matrix[(row * block_size + i) * N + (col * block_size + j)];
        }
    }
}

void blockMultiply(const std::vector<int>& A_block, const std::vector<int>& B_block, std::vector<int>& C_block, int block_size) {
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            for (int k = 0; k < block_size; ++k) {
                C_block[i * block_size + j] += A_block[i * block_size + k] * B_block[k * block_size + j];
            }
        }
    }
}

void mergeSubmatrix(std::vector<int>& C, const std::vector<int>& C_block, int row, int col, int block_size) {
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            C[(row * block_size + i) * N + (col * block_size + j)] = C_block[i * block_size + j];
        }
    }
}

void serialMultiply(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i * size + j] = 0;
            for (int k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int grid_size = static_cast<int>(sqrt(num_processes));
    if (grid_size * grid_size != num_processes) {
        std::cerr << "Error: The number of processes must be a perfect square." << std::endl;
        MPI_Finalize();
        return -1;
    }

    int block_size = N / grid_size;

    std::vector<int> A(N * N), B(N * N), C(N * N, 0);

    if (rank == 0) {
        srand(time(0));
        for (int i = 0; i < N * N; ++i) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }

        auto serial_start = std::chrono::high_resolution_clock::now();
        std::vector<int> serial_C(N * N, 0);
        serialMultiply(A, B, serial_C, N);
        auto serial_end = std::chrono::high_resolution_clock::now();
        auto serial_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serial_end - serial_start).count();
        std::cout << "Time for serial matrix multiplication: " << serial_duration << " ms" << std::endl;

        std::cout << "Matrix A:" << std::endl;
        displayMatrix(A, N);
        std::cout << "Matrix B:" << std::endl;
        displayMatrix(B, N);
    }

    MPI_Bcast(A.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm cart_comm, row_comm, col_comm;
    int cart_rank, coords[2];
    int dims[2] = {grid_size, grid_size};
    int periods[2] = {true, true};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, false, &cart_comm);
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    MPI_Comm_split(cart_comm, coords[0], coords[1], &row_comm);
    MPI_Comm_split(cart_comm, coords[1], coords[0], &col_comm);

    std::vector<int> A_block(block_size * block_size), B_block(block_size * block_size), C_block(block_size * block_size, 0);
    std::vector<int> temp_A_block(block_size * block_size);

    int row = coords[0];
    int col = coords[1];

    getSubmatrix(A, A_block, row, col, block_size);
    getSubmatrix(B, B_block, row, col, block_size);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Fox算法主循环
    for (int round = 0; round < grid_size; ++round) {
        // 广播当前行的A块
        std::copy(A_block.begin(), A_block.end(), temp_A_block.begin());
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(temp_A_block.data(), block_size * block_size, MPI_INT, (round + row) % grid_size, row_comm);
        MPI_Barrier(MPI_COMM_WORLD);

        // 执行块乘法
        blockMultiply(temp_A_block, B_block, C_block, block_size);

        // 将B矩阵上移
        MPI_Send(B_block.data(), block_size * block_size, MPI_INT, (row + 1) % grid_size, 0, col_comm);
        MPI_Recv(B_block.data(), block_size * block_size, MPI_INT, (row + grid_size - 1) % grid_size, 0, col_comm, MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank != 0) {
        MPI_Send(C_block.data(), block_size * block_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                if (i + j != 0) {
                    MPI_Recv(C_block.data(), block_size * block_size, MPI_INT, i * grid_size + j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                mergeSubmatrix(C, C_block, i, j, block_size);
            }
        }

        std::cout << "Matrix C after Fox algorithm:" << std::endl;
        displayMatrix(C, N);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();  // 以毫秒为单位
        std::cout << "Time for parallel matrix multiplication: " << duration << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
