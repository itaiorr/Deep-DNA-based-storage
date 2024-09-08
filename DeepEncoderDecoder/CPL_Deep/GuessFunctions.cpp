#include <set>
#include <algorithm>
#include <chrono>
#include "FreqFunctions.hpp"
#include "GuessFunctions.hpp"
#include "Graph.hpp"

// return index of string in strs1 which has minimum sum of edit distance from all strings in strs2
int MinSumEDIndex(const vector<string>& strs1, const vector<string>& strs2) {
	vector<int> sumEDs(strs1.size());
	for (unsigned i = 0; i < strs1.size(); i++) {
		sumEDs[i] = SumED(strs1[i], strs2);
	}
	int minIndex = min_element(sumEDs.begin(), sumEDs.end()) - sumEDs.begin();
	return minIndex;
}

string HeaviestPath(const Cluster& cluster, int index, const int correctLength, mt19937& generator) {
	vector<CondProb> mergedCum;
	ProbVec(index, cluster.copies, generator, mergedCum);
	GraphLD graph;
	vector<string> verticesToStrings;
	CondProbToStringsGraph(mergedCum, graph, verticesToStrings);
	string guess = HeaviestPathString(graph, verticesToStrings);
	return guess;
}

string HeaviestPathClosestOfTwo(const Cluster& cluster, int index, const int correctLength, mt19937& generator) {
	string guess1 = HeaviestPath(cluster, index, correctLength, generator);
	int lenDif1 = abs(((int) guess1.size()) - correctLength);
	string guess2 = HeaviestPath(cluster, index, correctLength, generator);
	int lenDif2 = abs(((int) guess2.size()) - correctLength);
	string guess3 = HeaviestPath(cluster, index, correctLength, generator);
	int lenDif3 = abs(((int) guess3.size()) - correctLength);
	if (lenDif1 <= lenDif2 and lenDif1 <= lenDif3) {
		return guess1;
	}
	else if (lenDif2 <= lenDif1 and lenDif2 <= lenDif3) {
		return guess2;
	}
	else {
		return guess3;
	}
}

// heaviest path of len correctLength in graph where each letter in string is a vertex and empty strings have no vertex
string FixedLenMarkov(const Cluster& cluster, int index, const int correctLength, mt19937& generator) {
	vector<CondProb> mergedCum;
	vector<char> letters;
	vector<vector<pair<int, long double>>> adj;
	vector<int> minLetters, maxLetters;
	ProbVec(index, cluster.copies, generator, mergedCum);
	GraphLD graph;
	CondProbToGraph(mergedCum, letters, adj, graph);
	MinMaxLettersToEnd(graph, minLetters, maxLetters);
	string guess = HighestScoreCommonStringMinMax(letters, adj, minLetters, maxLetters, correctLength);
	return guess;
}

string FixedLenMarkov(const Cluster& cluster, int index, const int correctLength, mt19937& generator, AlgTimes& times) {
	vector<CondProb> mergedCum;
	vector<char> letters;
	vector<vector<pair<int, long double>>> adj;
	vector<int> minLetters, maxLetters;

	times.StartProbVecTimer();
	ProbVec(index, cluster.copies, generator, mergedCum);
	times.StopProbVecTimer();

	GraphLD graph;

	times.StartGraphTimer();
	CondProbToGraph(mergedCum, letters, adj, graph);
	times.StopGraphTimer();

	times.StartMinMaxTimer();
	MinMaxLettersToEnd(graph, minLetters, maxLetters);
	times.StopMinMaxTimer();

	times.StartHeaviestPathTimer();
	string guess = HighestScoreCommonStringMinMax(letters, adj, minLetters, maxLetters, correctLength);
	times.StopHeaviestPathTimer();

	return guess;
}

// heaviest path of len correctLength in graph where each letter in string is a vertex and empty strings have no vertex
string FixedLenMarkovMaxShift(const Cluster& cluster, int index, const int correctLength, mt19937& generator,
		const int maxDiagLongDim, const int maxDiagShortDim) {
	vector<CondProb> mergedCum;
	vector<char> letters;
	vector<vector<pair<int, long double>>> adj;
	vector<int> minLetters, maxLetters;
	ProbVec(index, cluster.copies, generator, mergedCum, maxDiagLongDim, maxDiagShortDim);
	GraphLD graph;
	CondProbToGraph(mergedCum, letters, adj, graph);
	MinMaxLettersToEnd(graph, minLetters, maxLetters);
	string guess = HighestScoreCommonStringMinMax(letters, adj, minLetters, maxLetters, correctLength);
	return guess;
}

void CorrectClusterSimple(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFunc simpleGuessFunc, Cluster& correctedCluster) {
	int clusterSize = cluster.copies.size();
	correctedCluster.original = cluster.original;
	correctedCluster.copies.clear();
	correctedCluster.copies.resize(clusterSize);
	for (int copyIndex = 0; copyIndex < clusterSize; copyIndex++) {
		correctedCluster.copies[copyIndex] = simpleGuessFunc(cluster, copyIndex, correctLength, generator);
		if (correctedCluster.copies[copyIndex].empty()) {
			correctedCluster.copies[copyIndex] = HeaviestPath(cluster, copyIndex, correctLength, generator);
		}
	}
}

void CorrectClusterSimpleT(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFuncWTime simpleGuessFuncT, Cluster& correctedCluster, AlgTimes& times) {
	int clusterSize = cluster.copies.size();
	correctedCluster.original = cluster.original;
	correctedCluster.copies.clear();
	correctedCluster.copies.resize(clusterSize);
	for (int copyIndex = 0; copyIndex < clusterSize; copyIndex++) {
		correctedCluster.copies[copyIndex] = simpleGuessFuncT(cluster, copyIndex, correctLength, generator, times);
		if (correctedCluster.copies[copyIndex].empty()) {
			correctedCluster.copies[copyIndex] = HeaviestPath(cluster, copyIndex, correctLength, generator);
		}
	}
}

void CorrectClusterFixedLenMarkovMaxShift(const Cluster& cluster, const int correctLength, mt19937& generator,
		const int maxDiagLongDim, const int maxDiagShortDim, Cluster& correctedCluster) {
	int clusterSize = cluster.copies.size();
	correctedCluster.original = cluster.original;
	correctedCluster.copies.clear();
	correctedCluster.copies.resize(clusterSize);
	for (int copyIndex = 0; copyIndex < clusterSize; copyIndex++) {
		correctedCluster.copies[copyIndex] = FixedLenMarkovMaxShift(cluster, copyIndex, correctLength, generator,
				maxDiagLongDim, maxDiagShortDim);
		if (correctedCluster.copies[copyIndex].empty()) {
			correctedCluster.copies[copyIndex] = HeaviestPath(cluster, copyIndex, correctLength, generator);
		}
	}
}

string SimpleOnCorrectedCluster(const Cluster& cluster, int index, const int correctLength, mt19937& generator,
		SimpleGuessFunc simpleGuessFunc) {
	Cluster correctedCluster;
	CorrectClusterSimple(cluster, correctLength, generator, simpleGuessFunc, correctedCluster);
	return simpleGuessFunc(correctedCluster, index, correctLength, generator);
}

void CorrectClusterSimpleTwice(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFunc simpleGuessFunc, Cluster& correctedCluster) {
	Cluster correctedCluster1;
	CorrectClusterSimple(cluster, correctLength, generator, simpleGuessFunc, correctedCluster1);
	CorrectClusterSimple(correctedCluster1, correctLength, generator, simpleGuessFunc, correctedCluster);
}

void CorrectClusterSimpleTwiceT(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFuncWTime simpleGuessFuncT, Cluster& correctedCluster, AlgTimes& times1, AlgTimes& times2) {
	Cluster correctedCluster1;
	CorrectClusterSimpleT(cluster, correctLength, generator, simpleGuessFuncT, correctedCluster1, times1);
	CorrectClusterSimpleT(correctedCluster1, correctLength, generator, simpleGuessFuncT, correctedCluster, times2);
}

void CorrectClusterFixedLenMarkovMaxShiftTwice(const Cluster& cluster, const int correctLength, mt19937& generator,
		const int maxDiagLongDim, const int maxDiagShortDim, Cluster& correctedCluster) {
	Cluster correctedCluster1;
	CorrectClusterFixedLenMarkovMaxShift(cluster, correctLength, generator, maxDiagLongDim, maxDiagShortDim,
			correctedCluster1);
	CorrectClusterFixedLenMarkovMaxShift(correctedCluster1, correctLength, generator, maxDiagLongDim, maxDiagShortDim,
			correctedCluster);
}

string MinSumEDSimpleCorrectedClusterTwice(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFunc simpleGuessFunc) {
	Cluster correctedCluster;
	CorrectClusterSimpleTwice(cluster, correctLength, generator, simpleGuessFunc, correctedCluster);
	set<string> uniqueGuesses(correctedCluster.copies.begin(), correctedCluster.copies.end());
	vector<string> uniqueGuessesVec(uniqueGuesses.begin(), uniqueGuesses.end());
	int minIndex = MinSumEDIndex(uniqueGuessesVec, cluster.copies);
	return uniqueGuessesVec[minIndex];
}

string MinSumEDSimpleCorrectedClusterTwiceT(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFuncWTime simpleGuessFuncT, AlgTimes& times1, AlgTimes& times2) {
	Cluster correctedCluster;
	CorrectClusterSimpleTwiceT(cluster, correctLength, generator, simpleGuessFuncT, correctedCluster, times1, times2);
	set<string> uniqueGuesses(correctedCluster.copies.begin(), correctedCluster.copies.end());
	vector<string> uniqueGuessesVec(uniqueGuesses.begin(), uniqueGuesses.end());
	int minIndex = MinSumEDIndex(uniqueGuessesVec, cluster.copies);
	return uniqueGuessesVec[minIndex];
}

string MinSumEDFixedLenMarkovMaxShiftCorrectedClusterTwice(const Cluster& cluster, const int correctLength,
		mt19937& generator, const int maxDiagLongDim, const int maxDiagShortDim) {
	Cluster correctedCluster;
	CorrectClusterFixedLenMarkovMaxShiftTwice(cluster, correctLength, generator, maxDiagLongDim, maxDiagShortDim,
			correctedCluster);
	set<string> uniqueGuesses(correctedCluster.copies.begin(), correctedCluster.copies.end());
	vector<string> uniqueGuessesVec(uniqueGuesses.begin(), uniqueGuesses.end());
	int minIndex = MinSumEDIndex(uniqueGuessesVec, cluster.copies);
	return uniqueGuessesVec[minIndex];
}

string MinSumEDSimpleCorrectedClusterTwiceJoinR(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFunc simpleGuessFunc, const int R) {
	if (cluster.copies.size() == 1)
		return cluster.copies[0];
	Cluster correctedCluster;

	CorrectClusterSimpleTwice(cluster, correctLength, generator, simpleGuessFunc, correctedCluster);
	set<string> joinedUniqueGuesses(correctedCluster.copies.begin(), correctedCluster.copies.end());

	for (int i = 0; i < R - 1; i++) {
		CorrectClusterSimpleTwice(cluster, correctLength, generator, simpleGuessFunc, correctedCluster);
		joinedUniqueGuesses.insert(correctedCluster.copies.begin(), correctedCluster.copies.end());
	}
	vector<string> joinedGuesses(joinedUniqueGuesses.begin(), joinedUniqueGuesses.end());

	int minIndex = MinSumEDIndex(joinedGuesses, cluster.copies);
	return joinedGuesses[minIndex];
}
