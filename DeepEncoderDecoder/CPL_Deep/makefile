CXX = g++
CFLAGS = -std=c++17 -O3 -g3 -Wall -c -fmessage-length=0
RM = rm

# ****************************************************


main: DNAV8.o Cluster.o EditDistance.o FreqFunctions.o Graph.o GuessFunctions.o Utils.o
	$(CXX) -o main DNAV8.o Cluster.o EditDistance.o FreqFunctions.o Graph.o GuessFunctions.o Utils.o
# ****************************************************

DNAV8.o: DNAV8.cpp
	$(CXX) $(CFLAGS) -c DNAV8.cpp
	
Cluster.o: Cluster.cpp
	$(CXX) $(CFLAGS) -c Cluster.cpp

EditDistance.o: EditDistance.cpp
	$(CXX) $(CFLAGS) -c EditDistance.cpp

FreqFunctions.o: FreqFunctions.cpp
	$(CXX) $(CFLAGS) -c FreqFunctions.cpp

Graph.o: Graph.cpp
	$(CXX) $(CFLAGS) -c Graph.cpp

GuessFunctions.o: GuessFunctions.cpp
	$(CXX) $(CFLAGS) -c GuessFunctions.cpp

Utils.o: Utils.cpp
	$(CXX) $(CFLAGS) -c Utils.cpp

	


clean:
	$(RM) *.o

