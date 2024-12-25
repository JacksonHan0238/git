#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define TAG_WORKER_TO_SERVER 0
#define TAG_SERVER_TO_WORKER 1
#define MAX_ITERATIONS 1 

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    // 定义进程数量 N = P + Q
    int P = 2; 
    int Q = size - P; 

    int iteration = 0; 
    double serial_start, serial_end, parallel_start, parallel_end; 

    if (rank < Q) {
        int server_rank = Q + (rank % P); 
        srand(time(NULL) + rank); 

        while (iteration < MAX_ITERATIONS) {  
            // 1. 生成随机数并发送给对应的参数服务器
            double random_value = (double)rand() / RAND_MAX;

            parallel_start = MPI_Wtime();
            MPI_Send(&random_value, 1, MPI_DOUBLE, server_rank, TAG_WORKER_TO_SERVER, MPI_COMM_WORLD);
            printf("Worker %d sent value %f to server %d\n", rank, random_value, server_rank);

            // 2. 等待参数服务器返回更新后的值
            double updated_value;
            MPI_Recv(&updated_value, 1, MPI_DOUBLE, server_rank, TAG_SERVER_TO_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Worker %d received updated value %f from server %d\n", rank, updated_value, server_rank);
            parallel_end = MPI_Wtime();

            if (rank == 0) {  
                printf("Parallel computation time: %f seconds\n", parallel_end - parallel_start);
            }

            iteration++;  
        }
    } else {
        while (iteration < MAX_ITERATIONS) {  
            double sum = 0.0;
            int count = 0;

            serial_start = MPI_Wtime();

            for (int worker_rank = 0; worker_rank < Q; worker_rank++) {
                if (worker_rank % P == (rank - Q)) { 
                    double received_value;
                    MPI_Recv(&received_value, 1, MPI_DOUBLE, worker_rank, TAG_WORKER_TO_SERVER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    sum += received_value;
                    count++;
                    printf("Server %d received value %f from worker %d\n", rank - Q, received_value, worker_rank);
                }
            }

            serial_end = MPI_Wtime();

            // 2. 计算平均值
            double average = sum / count;
            printf("Server %d calculated average %f\n", rank - Q, average);

            // 3. 发送平均值给对应的所有工作进程
            for (int worker_rank = 0; worker_rank < Q; worker_rank++) {
                if (worker_rank % P == (rank - Q)) {
                    MPI_Send(&average, 1, MPI_DOUBLE, worker_rank, TAG_SERVER_TO_WORKER, MPI_COMM_WORLD);
                    printf("Server %d sent updated value %f to worker %d\n", rank - Q, average, worker_rank);
                }
            }

            if (rank == Q) { 
                printf("Serial computation time: %f seconds\n", serial_end - serial_start);
            }

            iteration++;  
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    MPI_Finalize();
    return 0;
}
