#include <iostream>
#include "mpi.h"
using namespace std;

int main(int argc, char** argv)
{
    int process_id, total_processes;
    int communication_tag = 1;
    MPI_Status mpi_status;
    
    MPI_Init(&argc, &argv);  

    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);  
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);   
    if (total_processes & (total_processes - 1)) {
        if (process_id == 0) {
            cout << "The number of processes should be a power of 2!" << endl;
        }
        MPI_Finalize();
        return 0;
    }

    int local_sum = process_id; 
    int received_data;

    double start_time = MPI_Wtime();

    for (int step_size = 1; step_size < total_processes; step_size *= 2) {
        int partner_id = process_id ^ step_size; 

        MPI_Sendrecv(&local_sum, 1, MPI_INT, partner_id, communication_tag, 
                     &received_data, 1, MPI_INT, partner_id, communication_tag, 
                     MPI_COMM_WORLD, &mpi_status);
        local_sum += received_data;
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

   /printf("Process ID: %d, Final Sum = %d\n", process_id, local_sum);

    double total_time;
    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (process_id == 0) {
        printf("Total Execution Time = %.6f seconds\n", total_time);
    }

    MPI_Finalize();  

    return 0;
}
