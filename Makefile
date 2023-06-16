INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcurand

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

# serial compiler
CC  = gcc

# flags
CFLAGS = -O3 -mavx -std=c99 -Wall -Wextra -pedantic
CFLAGS_OMP = -fopenmp
LIB2        = -lm
INCS =

all:		mc_pi_cuRAND mc_omp

mc_pi_cuRAND: 	mc_pi_cuRAND.cu Makefile
		nvcc mc_pi_cuRAND.cu -o mc_pi_cuRAND $(INC) $(NVCCFLAGS) $(LIB)

mc_omp:		mc_omp.c
		$(CC) $(INCS) $(CFLAGS) $(CFLAGS_OMP) -o mc_omp mc_omp.c $(LIB2) 

clean:
		rm -f mc_pi_cuRAND
		rm -f mc_omp
