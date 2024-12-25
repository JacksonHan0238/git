#include <iostream>
#include "mpi.h"
using namespace std;

int main(int argc, char *argv[])
{
    int rank, num_processes;
    int tag = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

    int local_sum = rank; 
    int received_data;
    MPI_Status status;

    double start_time = MPI_Wtime();

    for (int step_size = 1; step_size < num_processes; step_size *= 2) {
        int partner_rank = rank ^ step_size;  

        if (rank & step_size) {  
            MPI_Send(&local_sum, 1, MPI_INT, partner_rank, tag, MPI_COMM_WORLD);
        } else {  
            MPI_Recv(&received_data, 1, MPI_INT, partner_rank, tag, MPI_COMM_WORLD, &status);
            local_sum += received_data;  
        }
    }
    for (int step_size = num_processes / 2; step_size >= 1; step_size /= 2) {
        int partner_rank = rank ^ step_size;

        if (rank & step_size) {  
            MPI_Recv(&received_data, 1, MPI_INT, partner_rank, tag, MPI_COMM_WORLD, &status);
            local_sum += received_data;  
        } else { 
            MPI_Send(&local_sum, 1, MPI_INT, partner_rank, tag, MPI_COMM_WORLD);
        }
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    double total_time;
    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   printf("Process %d, Final Sum = %d\n", rank, local_sum);

    if (rank == 0) {
        printf("Total Execution Time = %.6f seconds\n", total_time);
    }

    MPI_Finalize(); 
    return 0;
}
