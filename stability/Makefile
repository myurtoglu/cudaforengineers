UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framework,GLUT
else
	LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
	LDFLAGS += -lglut -lGL -lGLU -lGLEW
endif

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations"

all: main.exe

main.exe: main.o kernel.o
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cpp kernel.h interactions.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
