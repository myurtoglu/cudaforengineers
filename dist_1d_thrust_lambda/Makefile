NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall --std=c++11 --expt-extended-lambda

main.exe: kernel.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o *.exe
