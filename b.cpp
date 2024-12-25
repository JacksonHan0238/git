#include <iostream>
#include "mpi.h"
using namespace std;

int main(int argc, char** argv)
{
    int pid, pnum;
    double starttime;
    MPI_Status status;
    int tag = 1;
    MPI_Init(&argc, &argv);  

    starttime = MPI_Wtime();  

    MPI_Comm_size(MPI_COMM_WORLD, &pnum);  
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);   

    int data = (pid + 1) * 2;  
    int recvdata;
    int dataa[pnum];

    int prev = (pid - 1 + pnum) % pnum;  
    int next = (pid + 1) % pnum;        


    MPI_Send(&data, 1, MPI_INT, next, tag, MPI_COMM_WORLD);  

    MPI_Recv(&recvdata, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, &status);  
    dataa[prev] = recvdata;  

    cout << pid << ": ";
    for (int i = 0; i < pnum; i++) {
        cout << dataa[i] << " ";
    }

    double elapsed_time = MPI_Wtime() - starttime;

    double total_time;
    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        printf("Total Execution Time = %.6lf s\n", total_time);
    }

    MPI_Finalize(); 

    return 0;
}
