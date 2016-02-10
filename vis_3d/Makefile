UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framework,GLUT
else
	LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
	LDFLAGS += -lglut -lGL -lGLU -lGLEW
endif

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS=-Xcompiler "-Wall -Wno-deprecated-declarations" -rdc=true
INC = -I/usr/local/cuda/samples/common/inc

all: main.exe

main.exe: main.o kernel.o device_funcs.o
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cpp kernel.h interactions.h
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

kernel.o: kernel.cu kernel.h device_funcs.cuh
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

device_funcs.o: device_funcs.cu device_funcs.cuh
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

clean:
	rm -f *.o *.exe
