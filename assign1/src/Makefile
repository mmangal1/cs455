CFLAGS = `pkg-config --libs --cflags opencv` -std=c++11

all: assign1

assign1: assign1.cpp
	g++ $(CFLAGS) assign1.cpp -o assign1

clean:
	rm assign1
