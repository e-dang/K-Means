#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <unistd.h>


int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int rank;
    int numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // std::cout << "RANK: " << rank << std::endl;
    // std::cout << "PROC: " << numProcs << std::endl;
    
    // int sum = 0;
    // int distances[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    // int local_sum = 0;
    // int chunk = 10 / numProcs;
    // int scrap = chunk + (10 % numProcs);
    // int start = chunk * rank;
    // int end = start + chunk - 1;
    // if(rank == (numProcs-1))
    // {
    //     end = start + scrap - 1;
    // }
    // // float f_start = 10.0 * (float(rank) / float(numProcs));
    // // int start = floor(f_start);
    // std::cout << "START: " << start << std::endl;
    // // float f_end = 10.0 * (float(rank + 1) / float(numProcs));
    // // int end = floor(f_end);
    // std::cout << "END: " << end << std::endl;
    // int local_distances[end-start];                                                                      
    // for (int pointIdx = 0; pointIdx <= (end-start); pointIdx++)
    // {
    //     local_distances[pointIdx] = pointIdx+start;
    //     local_sum += 1;
    //     std::cout << local_distances[pointIdx] << std::endl;
    // }
    // int recLen[numProcs];
    // int disp[numProcs];
    // for(int i = 0; i < numProcs; i++)
    // {
    //     recLen[i] = chunk;
    //     disp[i] = i * chunk;
    //     std::cout << "RECLEN" << recLen[i] << " DISP " << disp[i] << " \n"; //<< std::end;
    // }
    // std::cout << std::endl;
    // recLen[numProcs-1] = scrap;
    // MPI_Allgatherv(local_distances, (end - start + 1), MPI_INT, distances, recLen, disp, MPI_INT, MPI_COMM_WORLD);
    // // MPI_Allreduce(&local_sum, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // // std::cout << rank << std::endl;
    // std::cout << "[ ";
    // for (int i =0; i < 10; i++)
    // {
    //     std::cout << distances[i] << ", ";
    // }
    // std::cout << " ]" << std::endl;

    
    MPI_Win win;
    

    int drank = 0; //(rank == 0) ? 1 : 0;
    int* a;
    

    if(rank == 0)
    {
        // a = new int(10);
        MPI_Win_allocate_shared(10*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win);
        // MPI_Win_create(a,10*sizeof(int), sizeof(int),MPI_INFO_NULL,MPI_COMM_WORLD,&win);
        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
        int x = 1;
        a[0] = x;
        a[1] = x;
        a[2] = x;
        a[3] = x;
        a[4] = x;
        // MPI_Put(&x, 1, MPI_INT, drank, 0, 1, MPI_INT, win);
        // x++;
        // MPI_Put(&x, 1, MPI_INT, drank, 1, 1, MPI_INT, win);
        // x++;
        // MPI_Put(&x, 1, MPI_INT, drank, 2, 1, MPI_INT, win);
        // x++;
        // MPI_Put(&x, 1, MPI_INT, drank, 3, 1, MPI_INT, win);
        // x++;
        // MPI_Put(&x, 1, MPI_INT, drank, 4, 1, MPI_INT, win);
    }
    else
    {
        MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win);
        // MPI_Win_create(NULL, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
        int x = -1;
        MPI_Put(&x, 1, MPI_INT, drank, 5, 1, MPI_INT, win);
        x--;
        MPI_Put(&x, 1, MPI_INT, drank, 6, 1, MPI_INT, win);
        x--;
        MPI_Put(&x, 1, MPI_INT, drank, 7, 1, MPI_INT, win);
        x--;
        MPI_Put(&x, 1, MPI_INT, drank, 8, 1, MPI_INT, win);
        x--;
        MPI_Put(&x, 1, MPI_INT, drank, 9, 1, MPI_INT, win);
    }
    MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
    if(true)
    {
        std::cout << rank << " ";
        // std::cout << a[5] << " ";
        // for(int i = 0; i < 10; i++)
        // {
        //     int y = 7;
        //     int err = MPI_Get(&y, 1, MPI_INT, 0, i, 1, MPI_INT, win);
        //     std::cout << y << " ";
        // }
        int y[10] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
        int err = MPI_Get(&y, 10, MPI_INT, 0, 0, 10, MPI_INT, win);
        for(int i = 0; i < 10; i++)
        {
            std::cout << y[i] << " ";
        }
        std::cout << std::endl;
    }
 
    MPI_Win_free(&win);
    
    MPI_Finalize();
}