CXX = g++
LD = g++

CXXFLAGS = -Ofast -std=c++14 -mavx2 -mfma -fopenmp -ftree-vectorize -ffast-math -fopt-info-vec -Wall -I./include -I../data_types/include
LDFLAGS = -O2 -fopenmp

CXXFLAGS += -DRANDOM_SHUFFLE_STATE

all: bin/test.x

bin/test.x: obj/test.o
	$(LD) $(LDFLAGS) -o $@ $^

obj/test.o: src/test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f bin/* obj/* ./*~ src/*~ include/random/*~
