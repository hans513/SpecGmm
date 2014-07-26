
UNAME_S:=$(shell uname -s)
ifeq ($(UNAME_S),Linux)
	EIGEN=/home/hans/eigen
endif
ifeq ($(UNAME_S),Darwin)
	EIGEN=/Users/hans/eigen
endif

main: main.cpp D3Matrix.cpp Fastfood.cpp
	g++ -I $(EIGEN) -I $(EIGEN)/unsupported -std=c++11 -O3 -o specGmm main.cpp D3Matrix.cpp Fastfood.cpp
