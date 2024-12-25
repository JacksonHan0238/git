#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;
    MPI_Get_processor_name(hostname, &hostname_len);

    map<string, vector<int>> node_groups;

    vector<char> all_hostnames(size * MPI_MAX_PROCESSOR_NAME); 
    vector<int> all_ranks(size);

    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_hostnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&rank, 1, MPI_INT, all_ranks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            string node_hostname(&all_hostnames[i * MPI_MAX_PROCESSOR_NAME]);
            node_groups[node_hostname].push_back(all_ranks[i]);
        }

        for (const auto& entry : node_groups) {
            cout << "Node: " << entry.first << " has processes: ";
            for (int r : entry.second) {
                cout << r << " ";
            }
            cout << endl;
        }
    }

    MPI_Finalize();

    return 0;
}
