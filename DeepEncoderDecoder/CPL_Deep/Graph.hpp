
#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include <vector>
#include <stack>
#include "FreqFunctions.hpp"
using namespace std;

class GraphInt {
	int V; // No. of vertices'
	vector<vector<pair<int, int> > > graph;
	// A function used by longestPath
	void TopologicalSortUtil(int v, vector<bool>& visited, stack<int>& Stack);

public:
	GraphInt();
	GraphInt(int V); // Constructor

	// function to add an edge to graph
	void AddEdge(const int u, const int v, const int weight);
//	int Weight(const int u, const int v) const;
	int GetV() const;
//	void GetAdj(vector<vector<pair<int, int>>>& adj) const;
//	// Finds longest distances from given source vertex
//	vector<int> LongestPathLen(int s);
	vector<int> LongestPathDist(int s);
//	vector<int> ShortestPathLen(int s);
	vector<int> ShortestPathDist(int s);
//	vector<int> LongestPath(int u, int v);
};

class GraphLD {
	int V; // No. of vertices'
	vector<vector<pair<int, long double> > > graph;
	// A function used by longestPath
	void TopologicalSortUtil(int v, vector<bool>& visited, stack<int>& Stack);

public:
	GraphLD();
	GraphLD(int V); // Constructor

	// function to add an edge to graph
	void AddEdge(const int u, const int v, const long double weight);
//	long double Weight(const int u, const int v) const;
	int GetV() const;
	void GetAdj(vector<vector<pair<int, long double>>>& adj) const;
	// Finds longest distances from given source vertex
	vector<int> LongestPathLen(int s);
//	vector<int> LongestPathLenWithoutTopSort(int s);
	vector<int> LongestPath(int u, int v);
//	vector<int> LongestPathWithoutTopSort(int u, int v);
	void ReverseGraphWeight1(GraphInt& revGraph) const;
//	void ReverseGraphWeightStringLen(GraphInt& revGraph, const vector<string>& vertexesToStrings) const;
	void RemoveUnlikelyEdges(long double wtMin);
};

void CondProbToStringsGraph(const vector<CondProb>& prob, GraphLD& graph, vector<string>& vertexesToStrings);

void CondProbToGraph(const vector<CondProb>& prob, vector<char>& letters, vector<vector<pair<int, long double> > >& adj,
		GraphLD& graph);
void MinMaxLettersToEnd(const GraphLD& graph, vector<int>& minLetters, vector<int>& maxLetters);

string HighestScoreCommonStringMinMax(const vector<char>& letters1,
		const vector<vector<pair<int, long double> > >& adj1, const vector<int>& minLetters1,
		const vector<int>& maxLetters1, const int stringLen);

string HeaviestPathString(GraphLD& graph, const vector<string>& verticesToStrings);

#endif /* GRAPH_HPP_ */
