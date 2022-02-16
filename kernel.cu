
#include "common.h"
#include "timer.h"

#define TILE_DIM 32

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO


    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    int tx = threadIdx.x; int ty = threadIdx.y;
	


    __shared__ float A_s[TILE_DIM][TILE_DIM]; //Ms
    __shared__ float B_s[TILE_DIM][TILE_DIM]; //Ns

    float sum = 0.0f; //pValue


if(col < N &&row < M){
    for(unsigned int tile =0; tile <ceilf(K/(float)TILE_DIM) ++ tile){

        if(row < M && (tile*TILE_DIM + tx)<K)
			A_s[ty][tx] = A[row*K + tile*TILE_DIM + tx];
		else
			A_s[ty][tx] = 0;
		if(col < N && (tile*TILE_DIM + ty)<K)
			B_s[ty][tx] = B[(tile*TILE_DIM + ty)*N + col];
		else
			B_s[ty][tx] = 0;

		// after the entire tile's values are available, proceed
        __syncthreads();

        for(unsigned int i=0; i< TILE_DIM; ++i){
        sum += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
    }
    __syncthreads();
    }
    C[row*N + col] = sum;


}
}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, M*K*sizeof(float));
    cudaMalloc((void**) &B_d, K*N*sizeof(float));
    cudaMalloc((void**) &C_d, M*N*sizeof(float));




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(A_d, A, M*K*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(B_d, B, K*N*sizeof(float), cudaMemcpyHostToDevice );





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks((N + numThreadsPerBlock.x -1 )/numThreadsPerBlock.x, (M + numThreadsPerBlock.x -1 )/numThreadsPerBlock.x);
    mm_tiled_kernel <<< numBlocks, numThreadsPerBlock>>> (A_d, B_d, C_d,M,N,K);




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(C, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}

