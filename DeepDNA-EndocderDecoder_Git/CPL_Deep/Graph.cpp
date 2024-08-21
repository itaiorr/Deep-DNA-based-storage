#include <cassert>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include "Graph.hpp"

const int NINF = INT_MIN;
const int INF = INT_MAX;
const int NA = -1;

GraphLD::GraphLD() :
		V(), graph(V, vector<pair<int, long double>>()) {
}

GraphLD::GraphLD(int V) :
		V(V), graph(V, vector<pair<int, long double>>()) {
}

void GraphLD::AddEdge(const int u, const int v, const long double weight) {
	graph[u].push_back(make_pair(v, weight));
}

int GraphLD::GetV() const {
	return V;
}

void GraphLD::GetAdj(vector<vector<pair<int, long double>>>& adj) const {
	adj = graph;
}

void GraphLD::RemoveUnlikelyEdges(long double wtMin) {
	for (auto& adjv : graph) {
		for (vector<pair<int, long double> >::iterator it = adjv.begin(); it != adjv.end();) {
			if (it->second <= wtMin)
				it = adjv.erase(it);
			else
				it++;
		}
	}
}

// reverse all edges of graph. set all weights to 1
void GraphLD::ReverseGraphWeight1(GraphInt& revGraph) const {
	revGraph = GraphInt(V);
	for (int fromV = 0; fromV < V; fromV++) {
		for (vector<pair<int, long double>>::const_iterator prIt = graph[fromV].cbegin(); prIt != graph[fromV].cend();
				prIt++) {
			revGraph.AddEdge(prIt->first, fromV, 1);
		}
	}
}

void GraphLD::TopologicalSortUtil(int v, vector<bool>& visited, stack<int>& Stack) {
// Mark the current node as visited
	visited[v] = true;

// Recur for all the vertices adjacent to this vertex
	for (vector<pair<int, long double> >::iterator it = graph[v].begin(); it != graph[v].end(); it++) {
		if (not visited[it->first]) {
			TopologicalSortUtil(it->first, visited, Stack);
		}
	}

// Push current vertex to stack which stores topological sort
	Stack.push(v);
}

//const long double NINFLD = (long double) (-100000);

vector<int> GraphLD::LongestPathLen(int s) {
	stack<int> Stack;
	vector<bool> visited(V, false);
	vector<long double> dist(V, NINFLD);
	vector<int> parents(V, NA);

// Call the recursive helper function to store Topological
// Sort starting from all vertices one by one
	for (int i = 0; i < V; i++)
		if (visited[i] == false)
			// Note: s or i here? Answer: i and s are the same. because one call for Topological sort on s is enough.
			TopologicalSortUtil(i, visited, Stack);

// distance to source as 0
	dist[s] = 0;

// Process vertices in topological order
	while (Stack.empty() == false) {
		// Get the next vertex from topological order
		int u = Stack.top();
		Stack.pop();

		// Update distances of all adjacent vertices
		if (dist[u] != NINFLD) {
			for (vector<pair<int, long double> >::iterator it = graph[u].begin(); it != graph[u].end(); it++) {
				if (dist[it->first] < dist[u] + it->second) {
					dist[it->first] = dist[u] + it->second;
					parents[it->first] = u;
				}
			}
		}
	}
	return parents;
}

vector<int> GraphLD::LongestPath(int u, int v) {
	vector<int> path;
	vector<int> parents = LongestPathLen(u);
	if (parents[v] == NA) {
		return path;
	}
	else {
		path.push_back(v);
		int parent = parents[v];
		while (parent != NA) {
			path.insert(path.begin(), parent);
			parent = parents[parent];
		}
		return path;
	}
}

GraphInt::GraphInt() :
		V(), graph(V, vector<pair<int, int>>()) {
}

GraphInt::GraphInt(int V) :
		V(V), graph(V, vector<pair<int, int>>()) {
}

int GraphInt::GetV() const {
	return V;
}

void GraphInt::AddEdge(const int u, const int v, const int weight) {
	graph[u].push_back(make_pair(v, weight));
}

void GraphInt::TopologicalSortUtil(int v, vector<bool>& visited, stack<int>& Stack) {
// Mark the current node as visited
	visited[v] = true;

// Recur for all the vertices adjacent to this vertex
	for (vector<pair<int, int> >::iterator it = graph[v].begin(); it != graph[v].end(); it++) {
		if (not visited[it->first]) {
			TopologicalSortUtil(it->first, visited, Stack);
		}
	}

// Push current vertex to stack which stores topological sort
	Stack.push(v);
}

vector<int> GraphInt::ShortestPathDist(int s) {
	stack<int> Stack;
	vector<bool> visited(V, false);
	vector<int> dist(V, INF);
	vector<int> parents(V, NA);

// Call the recursive helper function to store Topological
// Sort starting from all vertices one by one
	for (int i = 0; i < V; i++)
		if (visited[i] == false)
			// Note: s or i here? Answer: i and s are the same. because one call for Topological sort on s is enough.
			TopologicalSortUtil(i, visited, Stack);

// distance to source as 0
	dist[s] = 0;

// Process vertices in topological order
	while (Stack.empty() == false) {
		// Get the next vertex from topological order
		int u = Stack.top();
		Stack.pop();

		// Update distances of all adjacent vertices
		if (dist[u] != INF) {
			for (vector<pair<int, int> >::iterator it = graph[u].begin(); it != graph[u].end(); it++) {
				if (dist[it->first] > dist[u] + it->second) {
					dist[it->first] = dist[u] + it->second;
					parents[it->first] = u;
				}
			}
		}
	}
	return dist;
}

vector<int> GraphInt::LongestPathDist(int s) {
	stack<int> Stack;
	vector<bool> visited(V, false);
	vector<int> dist(V, NINF);
	vector<int> parents(V, NA);

// Call the recursive helper function to store Topological
// Sort starting from all vertices one by one
	for (int i = 0; i < V; i++)
		if (visited[i] == false)
			// Note: s or i here? Answer: i and s are the same. because one call for Topological sort on s is enough.
			TopologicalSortUtil(i, visited, Stack);

// distance to source as 0
	dist[s] = 0;

// Process vertices in topological order
	while (Stack.empty() == false) {
		// Get the next vertex from topological order
		int u = Stack.top();
		Stack.pop();

		// Update distances of all adjacent vertices
		if (dist[u] != NINF) {
			for (vector<pair<int, int> >::iterator it = graph[u].begin(); it != graph[u].end(); it++) {
				if (dist[it->first] < dist[u] + it->second) {
					dist[it->first] = dist[u] + it->second;
					parents[it->first] = u;
				}
			}
		}
	}
	return dist;
}

long double TotalFrq(const CondFreq& condFreq) {
	long double totalFreq = 0;
	for (auto& strPairPr : condFreq) {
		totalFreq += strPairPr.second.second;
	}
	return totalFreq;
}

int CountStrings(const vector<CondFreq>& freqVec) {
	int stringNum = 0;
	for (auto& condFreq : freqVec) {
		stringNum += condFreq.size();
	}
	return stringNum;
}

int CountStrings(const vector<CondProb>& prob) {
	int stringNum = 0;
	for (auto& condProb : prob) {
		stringNum += condProb.size();
	}
	return stringNum;
}

int CountLetters(const vector<CondProb>& prob) {
	unsigned countLetters = 0;
	for (unsigned i = 0; i < prob.size(); i++) {
		for (auto& pr : prob[i]) {
			countLetters += pr.first.size();
		}
	}
	return int(countLetters);
}

// Convert a Cond Freq vector to a graph.
// The Graph:
// vertexes:	the main strings in CondProb vector plus start string and end string
// edges:		from start string to all first level main strings
//				from a main string to the strings in the next level referenced by its cond map
//				from main strings of last level to end string
// weights:		log(to string freq/ from string freq)

void CondProbToStringsGraph(const vector<CondProb>& prob, GraphLD& graph, vector<string>& vertexesToStrings) {
	int V = CountStrings(prob) + 2;
	graph = GraphLD(V);
	vertexesToStrings = vector<string>(V);
	vertexesToStrings[0] = START_STRING;
	vertexesToStrings[V - 1] = END_STRING;

	vector<map<string, int> > stringToVertex(prob.size());
	int vertex = 1;
	for (unsigned condProbIndex = 0; condProbIndex < prob.size(); condProbIndex++) {
		for (auto& pr : prob[condProbIndex]) {
			vertexesToStrings[vertex] = pr.first;
			stringToVertex[condProbIndex][pr.first] = vertex;
			vertex++;
		}
	}

	// edges from START_STRING to all first level strings with weight 0 because all toString|fromString of first prob are multiplied by fromString|START_STRING
	int fromVNum = 0;
	for (auto& strPairPr : prob[0]) {
		const string& toString = strPairPr.first;
		int toVNum = stringToVertex[0][toString];
		long double wt = 0; //
		graph.AddEdge(fromVNum, toVNum, wt);
	}

	for (unsigned condProbIndex = 0; condProbIndex < prob.size() - 1; condProbIndex++) {
		for (auto& strPairPr : prob[condProbIndex]) {
			const string& fromString = strPairPr.first;
			int fromVNum = stringToVertex[condProbIndex][fromString];
			for (auto& pr : strPairPr.second) {
				const string& toString = pr.first;
				int toVNum = stringToVertex[condProbIndex + 1][toString];
				long double wt = pr.second;
				graph.AddEdge(fromVNum, toVNum, wt);
			}
		}
	}
	// handle last condFreq
	for (auto& strPairPr : prob[prob.size() - 1]) {
		const string& fromString = strPairPr.first;
		int fromVNum = stringToVertex[prob.size() - 1][fromString];
		for (auto& pr : strPairPr.second) {
			int toVNum = V - 1;
			long double wt = pr.second;
			graph.AddEdge(fromVNum, toVNum, wt);
		}

	}

}

string HeaviestPathString(GraphLD& graph, const vector<string>& verticesToStrings) {
	vector<int> path = graph.LongestPath(0, graph.GetV() - 1);
	assert(not path.empty());
	path.erase(path.begin()); // remove start vertex
	path.pop_back(); // remove end vertex
	string guess;
	for (auto& vNum : path) {
		guess += verticesToStrings[vNum];
	}

	return guess;
}

void CondProbToGraph(const vector<CondProb>& prob, vector<char>& letters, vector<vector<pair<int, long double> > >& adj,
		GraphLD& graph) {
	int V = CountLetters(prob) + 2;
	graph = GraphLD(V);
	letters = vector<char>(V);
	map<string, int> lastLevelVs, currentLevelVs; // strings and the vertex number. what about inserts with multiple letters?
	vector<pair<int, long double>> carryVs; // vertexes reachable by empty string
	int vertexIndex = V - 1;
	letters[vertexIndex] = END_SYMBOL;
	lastLevelVs[string(1, END_SYMBOL)] = vertexIndex;
	vertexIndex--;
	// iterate in reverse
	for (vector<CondProb>::const_reverse_iterator rit = prob.rbegin(); rit != prob.rend(); rit++) {
		bool hasEmptyStr = false;
		for (auto& strMap : (*rit)) {
			string currentStr = strMap.first;
			const map<string, long double>& currentMap = strMap.second;
			int currentStrsize = currentStr.size();
			if (currentStrsize == 0) {
				hasEmptyStr = true;
			}
			else if (currentStrsize == 1) { // single letter string
				// connect to all non empty strings in map. if empty string in map, connect to carries. add to currentLevel
				letters[vertexIndex] = currentStr[0];
				for (auto& toPair : currentMap) {
					const string& toString = toPair.first;
					long double wt = toPair.second;
					if (toPair.first.empty()) { // to string is the empty string. connect to carries
						for (auto& pr : carryVs) {
							graph.AddEdge(vertexIndex, pr.first, pr.second + wt);
						}
					}
					else { // to string is not empty. connect to respective vertex
						auto findToStr = lastLevelVs.find(toString);
						assert(findToStr != lastLevelVs.end());
						graph.AddEdge(vertexIndex, findToStr->second, wt);
					}
				}
				currentLevelVs[currentStr] = vertexIndex;
				vertexIndex--;
			}
			else { 	// string size > 1. connect only the last letter to all non empty strings in map.
					// if empty string in map, connect last letter to carries. add first letter vertex to currentLevel

				letters[vertexIndex] = currentStr[currentStrsize - 1];
				for (auto& toPair : currentMap) {
					const string& toString = toPair.first;
					long double wt = toPair.second;
					if (toPair.first.empty()) { // to string is the empty string. connect to carries
						for (auto& pr : carryVs) {
							graph.AddEdge(vertexIndex, pr.first, pr.second + wt);
						}
					}
					else { // to string is not empty. connect to respective vertex
						auto findToStr = lastLevelVs.find(toString);
						assert(findToStr != lastLevelVs.end());
						graph.AddEdge(vertexIndex, findToStr->second, wt);
					}
				}

				vertexIndex--;
				for (int strIndex = currentStrsize - 2; strIndex >= 0; strIndex--) { // connect each letter, except last, to the one after with weight 0
					letters[vertexIndex] = currentStr[strIndex];
					graph.AddEdge(vertexIndex, vertexIndex + 1, 0);
					vertexIndex--;
				}
				// add first letter vertex (last in iteration) to currentLevelVs
				currentLevelVs[currentStr] = vertexIndex + 1;
			}
		}
		if (hasEmptyStr) { // level has empty string
			// if empty string has empty string in its map, add empty string weight to all existing carries. otherwise, clear carrys.
			auto emptyStrMapIt = rit->find("");
			assert(emptyStrMapIt != rit->end());
			auto findEmptyString = emptyStrMapIt->second.find("");
			if (findEmptyString == emptyStrMapIt->second.end()) {	// empty string doesn't have empty string in its map
				carryVs.clear();
			}
			else { // empty string has empty string in its map.
				   // add empty string weight to all existing carries
				for (auto& pr : carryVs) {
					pr.second += findEmptyString->second;
				}
			}

			// add new carries (strings in map of empty string) with their weight
			for (auto& strWtPair : emptyStrMapIt->second) {
				if (strWtPair.first.empty())	// don't add empty string to carry
					continue;
				auto findV = lastLevelVs.find(strWtPair.first);
				assert(findV != lastLevelVs.end());
				carryVs.push_back(make_pair(findV->second, strWtPair.second));
			}

		}
		else { // no empty string so nothing to carry
			carryVs.clear();
		}
		lastLevelVs = currentLevelVs;
		currentLevelVs.clear();
	}
	assert(vertexIndex == 0);
	letters[0] = END_SYMBOL;
	// connect start letter to all in first level
	long double wt = 0;
	for (auto& pr : lastLevelVs) {
		graph.AddEdge(0, pr.second, wt);
	}
	for (auto& pr : carryVs) {
		graph.AddEdge(0, pr.first, pr.second);
	}
	graph.GetAdj(adj);
}

// compute min and max edge path from each vertex to last vertex (V-1)
// by computing min max paths from last vertex to all vertexes in reversed graph.
void MinMaxPathsToEnd(const GraphLD& graph, vector<int>& minPaths, vector<int>& maxPaths) {
	GraphInt reverseGraph;
	graph.ReverseGraphWeight1(reverseGraph);

	minPaths = reverseGraph.ShortestPathDist(reverseGraph.GetV() - 1);
	maxPaths = reverseGraph.LongestPathDist(reverseGraph.GetV() - 1);
}

// edge paths in graph are to END_SYMBOL end vertex.
// letter path len is (not including path start letter):	for start vertex(0): edge path len - 1.
// 															for (1,V-2) vertexes: edge path len - 1.
//															for vertex V-1: 0;
void MinMaxLettersToEnd(const GraphLD& graph, vector<int>& minLetters, vector<int>& maxLetters) {
	vector<int> minPaths, maxPaths;
	int V = graph.GetV();
	minLetters = vector<int>(V);
	maxLetters = vector<int>(V);
	MinMaxPathsToEnd(graph, minPaths, maxPaths);
	for (int v = 0; v < V - 1; v++) {
		minLetters[v] = minPaths[v] - 1;
		maxLetters[v] = maxPaths[v] - 1;
	}
	minLetters[V - 1] = 0;
	maxLetters[V - 1] = 0;
}

struct Node {
	long double value = NINFLD;
	int parent = NA;
};

bool IsGoodVertex(const int V, const vector<int>& minLetters1, const vector<int>& maxLetters1, const int lettersSoFar,
		const int letterNumTarget) {

	if (V == (int) minLetters1.size() - 1) { // if vertex is end vertex it's ok
		return true;
	}
	int min = minLetters1[V];
	int max = maxLetters1[V];
	if (lettersSoFar + min > letterNumTarget) {
		return false;
	}
	if (lettersSoFar + max < letterNumTarget) {
		return false;
	}
	return true;
}

string HighestScoreCommonStringMinMax(const vector<char>& letters1,
		const vector<vector<pair<int, long double> > >& adj1, const vector<int>& minLetters1,
		const vector<int>& maxLetters1, const int stringLen) {
	string result;
	int K = stringLen + 1;
	vector<unordered_map<int, Node> > dp(K + 1);
	dp[0][0].value = 0;
	for (int i = 0; i < K; i++) {
		if (dp[i].empty()) {
			return result;
		}
		for (auto& mp : dp[i]) {
			const int fromV = mp.first;
			for (auto& edge : adj1[fromV]) {
				const int toV = edge.first;
				if (not IsGoodVertex(toV, minLetters1, maxLetters1, i + 1, stringLen)) { // letters so far: i plus 1 for toV
					continue;
				}
//				if (edge.second == NINFLD) seems to be redundant code. edge.second is never NINFLD
//					continue;
				Node& toVnode = dp[i + 1][toV];
				long double newWt = mp.second.value + edge.second;
				if (toVnode.value < newWt) {
					toVnode.value = newWt;
					toVnode.parent = fromV;
				}
			}
		}
	}

	// backtrack path
	int endVertex = adj1.size() - 1;
	unordered_map<int, Node>::iterator found = dp[K].find(endVertex);
	if (found == dp[K].end()) {
		return result;
	}
	Node node = found->second;
	for (int j = K; j > 1; j--) { // don't need first and last END_SYMBOL
		result.insert(result.begin(), letters1[node.parent]);
		node = dp[j - 1][node.parent];
	}
	return result;
}
