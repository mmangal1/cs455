CFLAGS = `pkg-config --libs --cflags opencv` -std=c++11

all: assign3

assign3: assign3.cpp
	g++ $(CFLAGS) assign3.cpp -o assign3

clean:
	rm assign3
