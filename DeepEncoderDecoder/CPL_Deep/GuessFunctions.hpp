#ifndef GUESSFUNCTIONS_HPP_
#define GUESSFUNCTIONS_HPP_

#include <iostream>
#include <iomanip>
#include "Cluster.hpp"

class AlgTimes {
	clock_t probVecStart, probVecEnd, totalProbVecTicks;
	clock_t graphStart, graphEnd, totalGraphTicks;
	clock_t minMaxStart, minMaxEnd, totalMinMaxTicks;
	clock_t heaviestPathStart, heaviestPathEnd, totalHeaviestPathTicks;
public:
	AlgTimes() :
			probVecStart(0), probVecEnd(0), totalProbVecTicks(0), graphStart(0), graphEnd(0), totalGraphTicks(0), minMaxStart(
					0), minMaxEnd(0), totalMinMaxTicks(0), heaviestPathStart(0), heaviestPathEnd(0), totalHeaviestPathTicks(
					0) {

	}
	void StartProbVecTimer() {
		probVecStart = clock();
	}
	void StopProbVecTimer() {
		probVecEnd = clock();
		totalProbVecTicks += (probVecEnd - probVecStart);
	}
	void StartGraphTimer() {
		graphStart = clock();
	}
	void StopGraphTimer() {
		graphEnd = clock();
		totalGraphTicks += (graphEnd - graphStart);
	}
	void StartMinMaxTimer() {
		minMaxStart = clock();
	}
	void StopMinMaxTimer() {
		minMaxEnd = clock();
		totalMinMaxTicks += (minMaxEnd - minMaxStart);
	}
	void StartHeaviestPathTimer() {
		heaviestPathStart = clock();
	}
	void StopHeaviestPathTimer() {
		heaviestPathEnd = clock();
		totalHeaviestPathTicks += (heaviestPathEnd - heaviestPathStart);
	}
	void PrintTimes() {
		cout << setprecision(2);
		cout << "Probability Vector Time:\t" << (double) totalProbVecTicks / CLOCKS_PER_SEC << endl;
		cout << "Graph Time:\t\t\t" << (double) totalGraphTicks / CLOCKS_PER_SEC << endl;
		cout << "MinMax Time:\t\t\t" << (double) totalMinMaxTicks / CLOCKS_PER_SEC << endl;
		cout << "Heaviest Path Time:\t\t" << (double) totalHeaviestPathTicks / CLOCKS_PER_SEC << endl;
	}
};

typedef string (*SimpleGuessFunc)(const Cluster& cluster, int index, const int correctLength, mt19937& generator);
typedef string (*SimpleGuessFuncWTime)(const Cluster& cluster, int index, const int correctLength, mt19937& generator,
		AlgTimes& times);

string HeaviestPath(const Cluster& cluster, int index, const int correctLength, mt19937& generator);
string HeaviestPathClosestOfTwo(const Cluster& cluster, int index, const int correctLength, mt19937& generator);
string FixedLenMarkov(const Cluster& cluster, int index, const int correctLength, mt19937& generator);


string FixedLenMarkov(const Cluster& cluster, int index, const int correctLength, mt19937& generator, AlgTimes& times);

string FixedLenMarkovMaxShift(const Cluster& cluster, int index, const int correctLength, mt19937& generator,
		const int maxDiagLongDim, const int maxDiagShortDim);

string SimpleOnCorrectedCluster(const Cluster& cluster, int index, const int correctLength, mt19937& generator,
		SimpleGuessFunc simpleGuessFunc);

string MinSumEDSimpleCorrectedClusterTwice(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFunc simpleGuessFunc);

string MinSumEDFixedLenMarkovMaxShiftCorrectedClusterTwice(const Cluster& cluster, const int correctLength,
		mt19937& generator, const int maxDiagLongDim, const int maxDiagShortDim);

string MinSumEDSimpleCorrectedClusterTwiceT(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFuncWTime simpleGuessFuncT, AlgTimes& times1, AlgTimes& times2);
string MinSumEDSimpleCorrectedClusterTwiceJoinR(const Cluster& cluster, const int correctLength, mt19937& generator,
		SimpleGuessFunc simpleGuessFunc, const int R);

#endif /* GUESSFUNCTIONS_HPP_ */
