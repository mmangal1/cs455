CFLAGS = `pkg-config --libs --cflags opencv` -std=c++11

all: assign2

assign2: assign2.cpp
	g++ $(CFLAGS) assign2.cpp -o assign2

clean:
	rm assign2
