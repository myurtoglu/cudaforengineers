NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
LIBS += -lcublas_static -lcusolver_static -lculibos

all: main.exe

main.exe: main.cpp
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

clean:
	rm -f *.o *.exe
