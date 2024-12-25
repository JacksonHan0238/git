#include <iostream>
#include <cstring>
#include "mpi.h"
using namespace std;

int main(int argc, char** argv) {
    int process_id, total_processes;
    int group_size = 3;
    int color, key;
    char message[16] = "a2";
    char received_message[16];
    MPI_Comm new_communicator;
    MPI_Status status;
    int message_tag = 1;

    double start_time, end_time;
    
    MPI_Init(&argc, &argv); 
    
    start_time = MPI_Wtime(); 
    
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);  

    color = process_id % group_size;
    key = process_id / group_size;
    
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_communicator);

    int split_rank, split_size;
    MPI_Comm_size(new_communicator, &split_size);
    MPI_Comm_rank(new_communicator, &split_rank);

    MPI_Request send_request[total_processes]; 
    MPI_Request recv_request[total_processes];  

    if (process_id == 0) {
        strcpy(received_message, message);
        for (int i = 1; i < total_processes && i < group_size; i++) {
            MPI_Isend(&message, 16, MPI_CHAR, i, message_tag, MPI_COMM_WORLD, &send_request[i]);  
        }
    } else if (process_id < group_size) {
        MPI_Irecv(&received_message, 16, MPI_CHAR, 0, message_tag, MPI_COMM_WORLD, &recv_request[process_id]);  
    }

    MPI_Bcast(&received_message, 16, MPI_CHAR, 0, new_communicator);  

    MPI_Waitall(total_processes, send_request, MPI_STATUSES_IGNORE);
    MPI_Waitall(total_processes, recv_request, MPI_STATUSES_IGNORE);

   // printf("New Communicator Rank %d, Original Rank %d, Group Size %d. The received message is %s\n", 
          // split_rank, process_id, split_size, received_message);

    end_time = MPI_Wtime(); 

    if (process_id == 0) {
        printf("Total execution time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize(); // Finalize MPI

    return 0;
}
