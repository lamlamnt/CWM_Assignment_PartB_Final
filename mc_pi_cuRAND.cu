#include <cuda.h>
#include <curand.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#define PI 3.14159265359

float calculate_accuracy(float pi);

__global__ void mc(float *d_x, float *d_y,int *d_area_count){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(d_x[index]*d_x[index] + d_y[index]*d_y[index] <= 1.0f){
		d_area_count[index] = 1;
	}
	else 
	{
		d_area_count[index] = 0;
	}
}

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
  }

  void Stop() {
    cudaEventRecord(stop, 0);
  }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

int main(void){
	//Try different number of randoms used
	int num_randoms[] = {10,100,1000,10000,100000,1000000,10000000,100000000};
	for(int j = 0; j < sizeof(num_randoms)/sizeof(num_randoms[0]); j++)
	{
		//Initialize the timer to calculate the speed 
		GpuTimer timer;
		timer.Start();
		printf("Number of randoms used %d \n", num_randoms[j]);
		
		//Initialize variables
   		int N = num_randoms[j];
		int area=0;	
	
		//Allocate pointers for host and device memory
		float *h_x; float *h_y;
		float *d_x; float *d_y;
		int *d_area_count;
		int *h_area_count;

		int mem_size = N*sizeof(float);
		h_x = (float*)malloc(mem_size);
		h_y = (float*)malloc(mem_size);
		h_area_count = (int*)malloc(mem_size);
		cudaMalloc((void**)&d_x,mem_size);
		cudaMalloc((void**)&d_y,mem_size);
		cudaMalloc((void**)&d_area_count,mem_size);
	
		//Declare variable
		curandGenerator_t gen;

		//Create random number generator
		curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);

		//Set the generator options
		curandSetPseudoRandomGeneratorSeed(gen,1234ULL);

		//Generate the randoms
		curandGenerateUniform(gen,d_x,N);
		curandGenerateUniform(gen,d_y,N);	
	
		mc<<<N/128,128>>>(d_x,d_y,d_area_count);

		//Copy result from device to host
		//cudaMemcpy(h_x,d_x,mem_size,cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_y,d_y,mem_size,cudaMemcpyDeviceToHost);
	
		//printf("Success %f %f %f",h_x[0], h_x[1], h_x[2]);
		cudaMemcpy(h_area_count,d_area_count,mem_size,cudaMemcpyDeviceToHost);
		for(int i = 0; i < N; i++){
			area += h_area_count[i];
		}
		float pi = (4*area)/float(N);
		printf("Pi is %f \n",pi);
		printf("The error is %f \n", calculate_accuracy(pi));
		//Clean up memory
		free(h_x);
		free(h_y);
		cudaFree(d_x);
		cudaFree(d_y);

		//Stop the timer
		timer.Stop();
		printf("Execution time: %f\n", timer.Elapsed()/10.0);
	}	
	return(0);	
}

float calculate_accuracy(float pi){
	return float(PI - pi);
}

