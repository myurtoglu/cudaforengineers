NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: main.exe

main.exe: main.o aux_functions.o
	$(NVCC) $^ -o $@

main.o: main.cpp aux_functions.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

aux_functions.o: aux_functions.cpp aux_functions.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
