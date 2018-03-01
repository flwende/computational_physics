CXX = g++
LD = g++

CXXFLAGS = -O3 -std=c++14 -mavx2 -mfma -fopenmp -faligned-new -ftree-vectorize -fopt-info-vec -ffast-math -Wall -I./include -I../random/include -I../data_types/include
LDFLAGS = -O2 -fopenmp

CXXFLAGS += -DRANDOM_SHUFFLE_STATE

all: bin/ising_2d.x

bin/ising_2d.x: obj/spin_system_2d.o obj/ising_2d.o
	$(LD) $(LDFLAGS) -o $@ $^

obj/ising_2d.o: src/ising_2d.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

obj/spin_system_2d.o: src/spin_system_2d.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f bin/* obj/* ./*~ src/*~ include/*~
