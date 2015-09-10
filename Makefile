CXX=g++
BIN=./bin

CXXFLAGS= -g -std=c++11 -pthread -lpthread -lgtest -lgtest_main -lglog
LOCAL_ROOT=third/local
THIRD_INCPATH=-I $(LOCAL_ROOT)/include
THIRD_LIB=-L $(LOCAL_ROOT)/lib

main: main.cpp
	mkdir -p $(BIN)
	$(CXX) main.cpp $(CXXFLAGS) $(THIRD_INCPATH) -Xlinker $(THIRD_LIB) -o $(BIN)/main.out
