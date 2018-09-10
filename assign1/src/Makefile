CFLAGS = `pkg-config --libs --cflags opencv` -std=c++11

all: q1

q1: q1.cpp
	g++ $(CFLAGS) q1.cpp -o q1

clean:
	rm q1
