NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

main.exe: main.cpp
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f *.o *.exe
