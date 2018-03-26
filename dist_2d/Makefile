NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: main.exe

main.exe: main.cu
	$(NVCC) $^ -o $@

clean:
	rm -f *.o *.exe
