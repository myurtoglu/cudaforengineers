NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
LIBS += -lnppi_static -lnppc_static -lculibos

all: main.exe

main.exe: main.cpp
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f *.o *.exe
