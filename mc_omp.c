#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//Function to calculate time
double wall_clock_time(void){
	#include <sys/time.h>
	#define MILLION 1000000.0
	double secs;
	struct timeval tp;
	gettimeofday(&tp,NULL);
	secs = (MILLION*(double) tp.tv_sec + (double) tp.tv_usec)/MILLION;
	return secs;	
}

int main() 
{
	//Loop through the different number of randoms used
	printf("\nUsing Open MP");
	int num_randoms[] = {10,100,1000,10000,100000,1000000,10000000,100000000};
	for(int j = 0; j < sizeof(num_randoms)/sizeof(num_randoms[0]); j++)
	{
		//Initialize variables
	 	int N=num_randoms[j];
	 	int area=0;
	 	int i = 0;
	 	float x;
   	 	float y;	
	 	double time_start, time_end;
	 
		//Start the clock
	 	time_start = wall_clock_time();

		//Use open mp to do parallel computing on the cpu
   	 	#pragma omp parallel shared(N,area) private(i,x,y)
   	 	{
			#pragma omp for reduction(+:area)	
    			for(i=0; i<N; i++) 
			{
				//Generate random numbers for x and y
        			x = ((float)rand())/RAND_MAX;
       	 			y = ((float)rand())/RAND_MAX;
				
				//Add to area count if point is inside the circle 
        			if(x*x + y*y <= 1.0f)
				{ 
					area++;
				}
    			}
    		}

		//Print the output and execution time
		printf("Number of randoms used: %d \n",num_randoms[j]);
		printf("\nPi using OpenMp:\t%f\n", (4.0*area)/(float)N);
	
		//Stop the clock and calculate execution time for code using CPU
		time_end = wall_clock_time();	
		printf("Execution time = %e s\n", time_end - time_start);
		return(0);
	}
}
