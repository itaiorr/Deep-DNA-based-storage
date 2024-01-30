#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <map>
#include <algorithm>
#include "Cluster.hpp"
#include "GuessFunctions.hpp"
#include "EditDistance.hpp"
#include "Utils.hpp"
using namespace std;

void PrintTestParam(const string& testName, const int testNum, const int strandLen, const int cloneNum, double delProb,
		double insProb, double subProb) {
	for (unsigned i = 0; i < testName.size(); i++)
		cout << "*";
	cout << endl;
	cout << testName << endl;
	cout << "---------------------------" << endl;
	cout << "Test Num:\t" << testNum << endl;
	cout << "Len:\t\t" << strandLen << endl;
	cout << "Clone Num:\t" << cloneNum << endl;
	cout << fixed << setprecision(2);
	cout << "Del Prob:\t" << delProb << endl;
	cout << "Ins Prob:\t" << insProb << endl;
	cout << "Sub Prob:\t" << subProb << endl;
	cout << "---------------------------" << endl;
}

void PrintTestResults(const double LER, const int noGuessNum, const double testTime, map<int, int>& histo) {
	cout << "LER:\t\t\t" << fixed << setprecision(8) << LER << endl;
	cout << "**********************************" << endl;
	cout << "Histogram:" << endl;
	int highestHistoVal = histo.rbegin()->first;
	int histoCum = 0;
	for (int i = 0; i <= highestHistoVal; i++) {
		histoCum += histo[i];
		cout << i << '\t' << histoCum << endl;
	}
	cout << "**********************************" << endl;
	cout << "No guess num:\t" << noGuessNum << endl;
	cout << "Test time:\t\t" << (int) testTime << " seconds" << endl;
	cout << "**********************************" << endl;
}

void GetCaseWithCopiesLimit(ifstream& input, string& original, vector<string>& copies2, const int maxCopies) {
    string line;
    original.clear();
    vector<string> copies;
    copies.clear();
    copies2.clear();
    getline(input, line);
    if (line.empty()) {
        return;
    }
    // first line is original
    original = line;

    // second line "*****" dump
    getline(input, line);

    // each line a copy until 2 empty lines.
    int endCase = 0;
    int copyIndex = 0;
    while (getline(input, line)) {
        if (line.empty()) {
            endCase++;
        }
        else{
            copies.push_back(line);
            copyIndex++;
        }
        if (endCase == 2) {
            break;
        }
    }

    //vector<int> sample;
    sample(copies.begin(), copies.end(),
                back_inserter(copies2),
                maxCopies,
                std::mt19937{std::random_device{}()});

}

void GetCaseWithCopiesLimitFirst(ifstream& input, string& original, vector<string>& copies, const int maxCopies) {
    string line;
    original.clear();
    copies.clear();
    getline(input, line);
    if (line.empty()) {
        return;
    }
    // first line is original
    original = line;

    // second line "*****" dump
    getline(input, line);

    // each line a copy until 2 empty lines.
    int endCase = 0;
    int copyIndex = 0;
    while (getline(input, line)) {
        if (line.empty()) {
            endCase++;
        }
        else {
            if (copyIndex < maxCopies) {
                copies.push_back(line);
                copyIndex++;
            }
        }
        if (endCase == 2) {
            break;
        }
    }
}

double TestSimple(int testNum, int strandLen, int cloneNum, double delProb, double insProb, double subProb, unsigned sd,
		SimpleGuessFunc simpleGuessFunc, const string& testName) {
	clock_t begin = clock();
	mt19937 clusterGenerator(sd), prioGenerator(sd);
	Cluster cluster;
	int cumED = 0;
	int noMutualCount = 0;
	int guessDistLTEOriginalDist = 0;
	map<int, int> histo;
	for (int i = 0; i < testNum; i++) {
		cluster = Cluster(strandLen, cloneNum, delProb, insProb, subProb, clusterGenerator);
		string guess = simpleGuessFunc(cluster, 0, strandLen, prioGenerator);

		int guessDist = SumED(guess, cluster.copies);
		int originalDist = SumED(cluster.original, cluster.copies);
		if (guessDist <= originalDist)
			guessDistLTEOriginalDist++;

		if (guess.empty()) {
			noMutualCount++;
			continue;
		}
		int currentED = EditDistance(guess, cluster.original);
		cumED += currentED;
		histo[currentED]++;
	}

	clock_t end = clock();
	double testTime = double(end - begin) / CLOCKS_PER_SEC;
	double noMutFreq = double(noMutualCount) / testNum;
	double LER = (double) cumED / ((testNum - noMutualCount) * strandLen);
	PrintTestParam(testName, testNum, strandLen, cloneNum, delProb, insProb, subProb);

	cout << "Guess Dist <= Original Dist:\t" << (double) guessDistLTEOriginalDist / (testNum - noMutualCount) << endl;
	cout << "---------------------------" << endl;
	PrintTestResults(LER, noMutFreq, testTime, histo);

	return LER;
}

double TestHeaviestPath(int testNum, int strandLen, int cloneNum, double delProb, double insProb, double subProb,
		unsigned sd) {
	string testName = "Heaviest Path Test";
	return TestSimple(testNum, strandLen, cloneNum, delProb, insProb, subProb, sd, HeaviestPath, testName);
}

double TestHeaviestPathClosestOfTwo(int testNum, int strandLen, int cloneNum, double delProb, double insProb,
		double subProb, unsigned sd) {
	string testName = "Heaviest Path Closest of Two Test";
	return TestSimple(testNum, strandLen, cloneNum, delProb, insProb, subProb, sd, HeaviestPathClosestOfTwo, testName);
}

double TestFixedLenMarkov(int testNum, int strandLen, int cloneNum, double delProb, double insProb, double subProb,
		unsigned sd) {
	string testName = "Fixed Len Markov Test";
	return TestSimple(testNum, strandLen, cloneNum, delProb, insProb, subProb, sd, FixedLenMarkov, testName);
}

double TestSimpleOnCorrectedCluster(int testNum, int strandLen, int cloneNum, double delProb, double insProb,
		double subProb, unsigned sd, SimpleGuessFunc simpleGuessFunc, const string& testName) {
	clock_t begin = clock();
	mt19937 clusterGenerator(sd), prioGenerator(sd);
	Cluster cluster;
	int cumED = 0;
	int noMutualCount = 0;
	int guessDistLTEOriginalDist = 0;
	map<int, int> histo;
	for (int i = 0; i < testNum; i++) {
		cluster = Cluster(strandLen, cloneNum, delProb, insProb, subProb, clusterGenerator);
		string guess = SimpleOnCorrectedCluster(cluster, 0, strandLen, prioGenerator, simpleGuessFunc);

		int guessDist = SumED(guess, cluster.copies);
		int originalDist = SumED(cluster.original, cluster.copies);
		if (guessDist <= originalDist)
			guessDistLTEOriginalDist++;

		if (guess.empty()) {
			noMutualCount++;
			continue;
		}
		int currentED = EditDistance(guess, cluster.original);
		cumED += currentED;
		histo[currentED]++;
	}

	clock_t end = clock();
	double testTime = double(end - begin) / CLOCKS_PER_SEC;
	double noMutFreq = double(noMutualCount) / testNum;
	double LER = (double) cumED / ((testNum - noMutualCount) * strandLen);
	PrintTestParam(testName, testNum, strandLen, cloneNum, delProb, insProb, subProb);

	cout << "Guess Dist <= Original Dist:\t" << (double) guessDistLTEOriginalDist / (testNum - noMutualCount) << endl;
	cout << "---------------------------" << endl;
	PrintTestResults(LER, noMutFreq, testTime, histo);

	return LER;
}

double TestFixedLenMarkovOnCorrectedCluster(int testNum, int strandLen, int cloneNum, double delProb, double insProb,
		double subProb, unsigned sd) {
	string testName = "Fixed Len Markov On Corrected Cluster Test";
	return TestSimpleOnCorrectedCluster(testNum, strandLen, cloneNum, delProb, insProb, subProb, sd, FixedLenMarkov,
			testName);
}

double TestMinSumEDSimpleCorrectedClusterTwice(int testNum, int strandLen, int cloneNum, double delProb, double insProb,
		double subProb, unsigned sd, SimpleGuessFunc simpleGuessFunc, const string& testName) {
	clock_t begin = clock();
	mt19937 clusterGenerator(sd), prioGenerator(sd);
	Cluster cluster;
	int cumED = 0;
	int noMutualCount = 0;
	map<int, int> histo;
	for (int i = 0; i < testNum; i++) {
		cluster = Cluster(strandLen, cloneNum, delProb, insProb, subProb, clusterGenerator);
		string mutGuess = MinSumEDSimpleCorrectedClusterTwice(cluster, strandLen, prioGenerator, simpleGuessFunc);
		if (mutGuess.empty()) {
			noMutualCount++;
			continue;
		}
		int currentED = EditDistance(mutGuess, cluster.original);
		cumED += currentED;
		histo[currentED]++;
	}

	clock_t end = clock();
	double testTime = double(end - begin) / CLOCKS_PER_SEC;
	double noMutFreq = double(noMutualCount) / testNum;
	double LER = (double) cumED / ((testNum - noMutualCount) * strandLen);
	PrintTestParam(testName, testNum, strandLen, cloneNum, delProb, insProb, subProb);

	cout << "---------------------------" << endl;
	PrintTestResults(LER, noMutFreq, testTime, histo);

	return LER;
}

double TestMinSumFixedLenMarkovCorrectedClusterTwice(int testNum, int strandLen, int cloneNum, double delProb,
		double insProb, double subProb, unsigned sd) {
	string testName = "Min Sum Fixed Len Corrected Cluster Twice Test";
	return TestMinSumEDSimpleCorrectedClusterTwice(testNum, strandLen, cloneNum, delProb, insProb, subProb, sd,
			FixedLenMarkov, testName);
}

double TestMinSumHeaviestPathCorrectedClusterTwice(int testNum, int strandLen, int cloneNum, double delProb,
		double insProb, double subProb, unsigned sd) {
	string testName = "Min Sum Heaviest Path Corrected Cluster Twice Test";
	return TestMinSumEDSimpleCorrectedClusterTwice(testNum, strandLen, cloneNum, delProb, insProb, subProb, sd,
			HeaviestPath, testName);
}

double TestMinSumEDSimpleCorrectedClusterTwiceT(int testNum, int strandLen, int cloneNum, double delProb,
		double insProb, double subProb, unsigned sd, SimpleGuessFuncWTime simpleGuessFuncT, const string& testName) {
	clock_t begin = clock();
	mt19937 clusterGenerator(sd), prioGenerator(sd);
	Cluster cluster;
	int cumED = 0;
	int noMutualCount = 0;
	map<int, int> histo;
	AlgTimes times1, times2;
	for (int i = 0; i < testNum; i++) {
		cluster = Cluster(strandLen, cloneNum, delProb, insProb, subProb, clusterGenerator);
		string mutGuess = MinSumEDSimpleCorrectedClusterTwiceT(cluster, strandLen, prioGenerator, simpleGuessFuncT,
				times1, times2);
		if (mutGuess.empty()) {
			noMutualCount++;
			continue;
		}
		int currentED = EditDistance(mutGuess, cluster.original);
		cumED += currentED;
		histo[currentED]++;
	}

	clock_t end = clock();
	double testTime = double(end - begin) / CLOCKS_PER_SEC;
	double noMutFreq = double(noMutualCount) / testNum;
	double LER = (double) cumED / ((testNum - noMutualCount) * strandLen);
	PrintTestParam(testName, testNum, strandLen, cloneNum, delProb, insProb, subProb);

	cout << "---------------------------" << endl;
	PrintTestResults(LER, noMutFreq, testTime, histo);

	cout << "Alg times:" << endl;
	cout << "==========" << endl;
	cout << "First Correction:" << endl;
	cout << "-----------------" << endl;
	times1.PrintTimes();
	cout << "Second Correction:" << endl;
	cout << "-----------------" << endl;
	times2.PrintTimes();

	return LER;
}

double TestMinSumFixedLenMarkovCorrectedClusterTwiceT(int testNum, int strandLen, int cloneNum, double delProb,
		double insProb, double subProb, unsigned sd) {
	string testName = "Min Sum Fixed Len Corrected Cluster Twice Test";
	return TestMinSumEDSimpleCorrectedClusterTwiceT(testNum, strandLen, cloneNum, delProb, insProb, subProb, sd,
			FixedLenMarkov, testName);
}

double TestMinSumEDFixedLenMarkovMaxShiftCorrectedClusterTwice(int testNum, int strandLen, int cloneNum, double delProb,
		double insProb, double subProb, unsigned sd, int maxDiagLongDim, int maxDiagShortDim) {
	clock_t begin = clock();
	mt19937 clusterGenerator(sd), prioGenerator(sd);
	Cluster cluster;
	int cumED = 0;
	int noMutualCount = 0;
	map<int, int> histo;
	for (int i = 0; i < testNum; i++) {
		cluster = Cluster(strandLen, cloneNum, delProb, insProb, subProb, clusterGenerator);
		string mutGuess = MinSumEDFixedLenMarkovMaxShiftCorrectedClusterTwice(cluster, strandLen, prioGenerator,
				maxDiagLongDim, maxDiagShortDim);
		if (mutGuess.empty()) {
			noMutualCount++;
			continue;
		}
		int currentED = EditDistance(mutGuess, cluster.original);
		cumED += currentED;
		histo[currentED]++;
	}

	clock_t end = clock();
	string testName = "Min Sum Fixed Len Max Shift Corrected Cluster Twice Test";
	double testTime = double(end - begin) / CLOCKS_PER_SEC;
	double noMutFreq = double(noMutualCount) / testNum;
	double LER = (double) cumED / ((testNum - noMutualCount) * strandLen);
	PrintTestParam(testName, testNum, strandLen, cloneNum, delProb, insProb, subProb);

	cout << "---------------------------" << endl;
	PrintTestResults(LER, noMutFreq, testTime, histo);

	return LER;
}

void TestDiagSquare(int testNum, unsigned sd) {
	mt19937 generator(sd);
	uniform_int_distribution<int> distNums(1, 1000);
	uniform_int_distribution<int> distMSize(2, 30);
	for (int test = 0; test < testNum; test++) {
		int m = distMSize(generator);
		int n = distMSize(generator);
		vector<vector<int> > M(m, vector<int>(n));
		uniform_int_distribution<int> distDiagLower(1, m - 1);
		uniform_int_distribution<int> distDiagUpper(1, n - 1);
		int uppermostDiag = distDiagUpper(generator);
		int lowermostDiag = distDiagLower(generator);

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (i - j > lowermostDiag || j - i > uppermostDiag)
					continue;
				M[i][j] = distNums(generator);
			}
		}

		DiagMat D(m, n, uppermostDiag, lowermostDiag, 0);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (i - j > lowermostDiag || j - i > uppermostDiag)
					continue;
				D(i, j) = M[i][j];
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (i - j > lowermostDiag || j - i > uppermostDiag)
					continue;
				if (D(i, j) != M[i][j]) {
					cout << "Test Failure" << endl;
					return;
				}
			}
		}
	}
	cout << "Test Success" << endl;
}

double TestMinSumEDSimpleCorrectedClusterTwiceJoinR(int testNum, int strandLen, int cloneNum, const double delProb,
		const double insProb, const double subProb, unsigned sd, const int R, SimpleGuessFunc simpleGuessFunc,
		const string& testName) {
	clock_t begin = clock();
	mt19937 clusterGenerator(sd), prioGenerator(sd);
	Cluster cluster;
	int cumED = 0;
	int noMutualCount = 0;
	int guessDistLTEOriginalDist = 0;
	map<int, int> histo;
	for (int i = 0; i < testNum; i++) {
		cluster = Cluster(strandLen, cloneNum, delProb, insProb, subProb, clusterGenerator);
		string guess = MinSumEDSimpleCorrectedClusterTwiceJoinR(cluster, strandLen, prioGenerator, simpleGuessFunc, R);

		int guessDist = SumED(guess, cluster.copies);
		int originalDist = SumED(cluster.original, cluster.copies);
		if (guessDist <= originalDist)
			guessDistLTEOriginalDist++;

		if (guess.empty()) {
			noMutualCount++;
			continue;
		}
		int currentED = EditDistance(guess, cluster.original);
		cumED += currentED;
		histo[currentED]++;
	}

	clock_t end = clock();
	double testTime = double(end - begin) / CLOCKS_PER_SEC;
	double LER = (double) cumED / ((testNum - noMutualCount) * strandLen);
	PrintTestParam(testName, testNum, strandLen, cloneNum, delProb, insProb, subProb);
	cout << "Rounds Num:\t" << R << endl;
	cout << "Guess Dist <= Original Dist:\t" << (double) guessDistLTEOriginalDist / (testNum - noMutualCount) << endl;
	cout << "---------------------------" << endl;
	PrintTestResults(LER, noMutualCount, testTime, histo);

	return LER;
}

double TestMinSumFixedLenMarkovCorrectedClusterTwiceJoinR(int testNum, int strandLen, int cloneNum,
		const double delProb, const double insProb, const double subProb, unsigned sd, const int R) {
	string testName = "Min Sum Fixed Len Corrected Cluster Twice Join R rounds Test";
	return TestMinSumEDSimpleCorrectedClusterTwiceJoinR(testNum, strandLen, cloneNum, delProb, insProb, subProb, sd, R,
			FixedLenMarkov, testName);
}

// The original - a string over the alphabet {A,C,G,T}
// strands - the corrupt copies of the original
// originalLen - length of the original string
// R - R parameter of the algorithm. default value: 1
// return value - the guess of CPL algorithm

// Compilation example
//g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o "src\\DNAV8.o" "..\\src\\DNAV8.cpp"
//g++ -o DNAV8.exe "src\\Cluster.o" "src\\DNAV8.o" "src\\EditDistance.o" "src\\FreqFunctions.o" "src\\Graph.o" "src\\GuessFunctions.o" "src\\Utils.o"


string CPLGuess(const vector<string>& strands, int originalLen, int R=1) {
	string original; // dummy variable. will not be used in this context
	unsigned sd = chrono::high_resolution_clock::now().time_since_epoch().count();
	mt19937 generator(sd);
	Cluster cluster(original, strands);
	return MinSumEDSimpleCorrectedClusterTwiceJoinR(cluster, originalLen, generator, FixedLenMarkov, R);
}


void TestFromFile(const string& inputFilename, int testNum, int strandLen, int maxCopies, int delPatternLen,
        const int subPriority, const int delPriority, const int insPriority, const int maxReps) {
    ifstream input;
    input.open(inputFilename.c_str());
    if (!input.is_open()) {
        cout << "Failed opening input file!" << endl;
        return;
    }
    int R=1;
    string original;
    vector<string> copies;
    unsigned sd = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 generator(sd);
    int originalLen = original.length();
    int cumTotalFinalGuessEditDist = 0, roundFinalGuessEditDist = 0;
    int cumFinalGuessSubstitutions = 0, cumFinalGuessInsertions = 0, cumFinalGuessDeletions = 0;
    map<int, int> editDistanceHist;

    int countFiltered = 0;
    for (int i = 1; i <= testNum; i++) {
        GetCaseWithCopiesLimit(input, original, copies, maxCopies);

        if (original.empty()) {
            break;
        }
        Cluster cluster(original, copies);
        // computing the avg length of the sequence;
        double avglen=0;
        for(string &cp: copies){
            avglen+=cp.length();
        }
        avglen=avglen/copies.size();

        string finalGuess;
        if (copies.size() == 1) { // if only 1 copy return copy.
            finalGuess = copies[0];
        }
        else {
            finalGuess = MinSumEDSimpleCorrectedClusterTwiceJoinR(cluster, originalLen, generator, FixedLenMarkov, R);
        }
        roundFinalGuessEditDist = EditDistance(original, finalGuess);
        editDistanceHist[roundFinalGuessEditDist]++;
        cumTotalFinalGuessEditDist += roundFinalGuessEditDist;

				//vector<LetterOps> result = ComputeEditDistancePriority(finalGuess, original, 0, generator);
        //map<string, double> countOperations = CountOperations(result);
        //assert(countOperations["I"] + countOperations["D"] + countOperations["R"] == roundFinalGuessEditDist);

       //cumFinalGuessSubstitutions += countOperations["R"];
       //cumFinalGuessInsertions += countOperations["I"];
       //cumFinalGuessDeletions += countOperations["D"];
    }
    map<int, int>::reverse_iterator rit = editDistanceHist.rbegin(); // points to last element in map
    int highestED = rit->first;
    int cumDist = 0;
    cout << "Edit distance hist:" << endl;
    for (int i = 0; i <= highestED; i++) {
        cumDist += editDistanceHist[i];
        cout << i << "\t" << cumDist << endl;
    }
    cout << "Number of filtered:\t" << countFiltered << endl;
    cout << "Avg. guess substitutions:\t"
            << 1000 * (double) cumFinalGuessSubstitutions / ((testNum - countFiltered) * strandLen) << endl;
    cout << "Avg. guess deletions:\t"
            << 1000 * (double) cumFinalGuessDeletions / ((testNum - countFiltered) * strandLen) << endl;
    cout << "Avg. guess insertions:\t"
            << 1000 * (double) cumFinalGuessInsertions / ((testNum - countFiltered) * strandLen) << endl;
    cout << "Avg. guess edit dist:\t"
            << 1000 * (double) cumTotalFinalGuessEditDist / ((testNum - countFiltered) * strandLen) << endl;

    input.close();
}


int getTestNum(const string& inputFilename){
    ifstream input;
    input.open(inputFilename.c_str());
    if (!input.is_open()) {
        cout << "Failed opening input file!" << endl;
        return -1;
    }
    string line;
    int count=0;
    while (getline(input, line)) {
        if (line[0]=='*') {
            count++;
        }
    }
    return count;
}

void GetAllCopies(ifstream& input, string& original, vector<string>& copies) {
    string line;
    original.clear();
    copies.clear();
    getline(input, line);
    if (line.empty()) {
        return;
    }
    // first line is original
    original = line;

    // second line "*****" dump
    getline(input, line);

    // each line a copy until 2 empty lines.
    int endCase = 0;
    while (getline(input, line)) {
        if (line.empty()) {
            endCase++;
        }
        else {
            copies.push_back(line);
        }

        if (endCase == 2) {
            break;
        }
    }
}

void AdvanceFile(ifstream& input, const int caseNum) {
    string original;
    vector<string> copies;
    for (int i = 0; i < caseNum; i++) {
        GetAllCopies(input, original, copies);
    }
}

// counting from 1
// test [startCase,endCase]
void TestFromFileCaseRange(const string& inputFilename, const string& outputFilename,
                           const string& resultsFilename_success, const string& resultsFilename_fail,
                           int startCase, int endCase,
        int strandLen, int maxCopies, int delPatternLen, const int subPriority, const int delPriority,
        const int insPriority, const int maxReps) {
    ifstream input;
    input.open(inputFilename.c_str());
    if (!input.is_open()) {
        cout << "Failed opening input file!" << endl;
        return;
    }
    int R=1;
    ofstream output;
    output.open(outputFilename.c_str());
    if (not output.is_open()) {
        cout << "Error opening output file!" << endl;
        return;
    }

    ofstream results_success;
    results_success.open(resultsFilename_success.c_str());
    if (not results_success.is_open()) {
        cout << "Error opening results file!" << endl;
        return;
    }

    ofstream results_fail;
    results_fail.open(resultsFilename_fail.c_str());
    if (not results_fail.is_open()) {
        cout << "Error opening results file!" << endl;
        return;
    }

    string original;
    vector<string> copies;
    int casesToAdvance = startCase - 1;
    AdvanceFile(input, casesToAdvance);
    int testNum = endCase - startCase + 1;
    unsigned sd = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 generator(sd);
    int roundFinalGuessEditDist = 0;
    double cumTotalFinalGuessEditDist = 0;
    double cumFinalGuessSubstitutions = 0, cumFinalGuessInsertions = 0, cumFinalGuessDeletions = 0;
    double error_rate=0.0;
    map<int, int> editDistanceHist;
    int majoCounter=0;
    for (int i = 1; i <= testNum; i++) {
        cout << int(100*(double(i)/testNum)) << endl;
        GetCaseWithCopiesLimit(input, original, copies, maxCopies);
        if (original.empty()) {
            break;
        }
        Cluster cluster(original, copies);
        // computing the avg length of the sequence;
        double avglen=0;
        for(string &cp: copies){
            avglen+=cp.length();
        }
        avglen=avglen/copies.size();

        string finalGuess;
        if (copies.size() == 1) { // if only 1 copy return copy.
            finalGuess = copies[0];
        }
        else {
            int originalLen =original.length();
            finalGuess = MinSumEDSimpleCorrectedClusterTwiceJoinR(cluster, originalLen, generator, FixedLenMarkov, R);

        }


        roundFinalGuessEditDist = EditDistance(original, finalGuess);
        editDistanceHist[roundFinalGuessEditDist]++;
        cumTotalFinalGuessEditDist += roundFinalGuessEditDist;
        if(roundFinalGuessEditDist>0){
            results_fail << "Cluster Num: " << i << endl;
            results_fail << original << endl;
            results_fail << finalGuess << endl;
            results_fail << "Distance: " << roundFinalGuessEditDist << endl << endl;

        }
        else{
            results_success << "Cluster Num: " << i << endl;
            results_success << original << endl;
            results_success << finalGuess << endl;
            results_success << "Distance: " << roundFinalGuessEditDist << endl << endl;

        }

       /* vector<LetterOps> result = ComputeEditDistancePriority(finalGuess, cluster.Original(), 0, generator);
        map<string, double> countOperations = CountOperations(result);
        assert(countOperations["I"] + countOperations["D"] + countOperations["R"] == roundFinalGuessEditDist);

        if(original.length()){
            cumTotalFinalGuessEditDist =((i-1)*cumTotalFinalGuessEditDist+double(roundFinalGuessEditDist)/(original.length() ))/i;
            cumFinalGuessSubstitutions =((i-1)*cumFinalGuessSubstitutions+(countOperations["R"])/(original.length() ))/i;
            cumFinalGuessInsertions =((i-1)*cumFinalGuessInsertions+(countOperations["I"])/(original.length() ))/i;
            cumFinalGuessDeletions =((i-1)*cumFinalGuessDeletions+((countOperations["D"])/(original.length())))/i;
            error_rate= ((i-1) *error_rate +(double(roundFinalGuessEditDist)/(original.length())))/i;
        }*/
    }
    cout << "StartCase:\t" << startCase << endl;
    cout << "EndCase:\t" << endCase << endl;
    output << "StartCase:\t" << startCase << endl;
    output << "EndCase:\t" << endCase << endl;
    map<int, int>::reverse_iterator rit = editDistanceHist.rbegin(); // points to last element in map
    int highestED = rit->first;
    int cumDist = 0;
    cout << "Total number of clusters: " << testNum-1 << endl;
    output << "Total number of clusters: " << testNum-1 << endl;

    cout << "Edit distance hist:" << endl;
    output << "Edit distance hist:" << endl;
    for (int i = 0; i <= highestED; i++) {
        cumDist += editDistanceHist[i];
        cout << i << "\t" << cumDist << endl;
        output << i << "\t" << cumDist << endl;
    }

    cout << "Substitution rate:\t" << cumFinalGuessSubstitutions << endl;
    cout << "Deletion rate:\t" << cumFinalGuessDeletions << endl;
    cout << "Insertion rate:\t" << cumFinalGuessInsertions << endl;
    //cout << "guess edit dist:\t" << cumTotalFinalGuessEditDist << endl;
    cout << "Error rate:\t" << error_rate << endl;
    cout << "Success rate:\t" << (double)(editDistanceHist[0])/testNum << endl;
    cout << "number of majority test " << majoCounter << endl;

    output << "Substitution rate:\t" << cumFinalGuessSubstitutions << endl;
    output << "Deletion rate:\t" << cumFinalGuessDeletions << endl;
    output << "Insertion rate:\t" << cumFinalGuessInsertions << endl;
    //output << "guess edit dist:\t" << cumTotalFinalGuessEditDist << endl;
    output << "Error rate:\t" << error_rate << endl;
    output << "Success rate:\t" << (double)(editDistanceHist[0])/testNum << endl;
    output << "number of majority test " << majoCounter << endl;
    input.close();
    output.close();
}

// main function
int main(int argc, char *argv[])
{
    if(argc<3){
        cout<< "not enough argument" << endl;
        return 1;
    }

    string input_file = argv[1];
    string output_path = argv[2];

    clock_t begin = clock();

    int maxCopies = 32;
//    set<int> filter = { 4490, 4756, 4850, 4879, 4896, 4929, 4937, 4940, 4950, 4955, 4969, 4971, 4973, 4977, 4978, 4979,
//            4981, 4983, 4985, 4986, 4987 };
//    CasesStats("evyaR.txt", maxCopies, EDThresh, filter); // second batch: strand length 117. casesNum 4989. Luis: len 150, num 596499

//    CasesFromFile("evyaLuis.txt",maxCopies);

//    int testNum = 1000;

//    int strandLen = 100;
//    int cloneNum = 20;

    int delPatternLen = 3;

    int subPriority = 0;
    int delPriority = 0;
    int insPriority = 0;

    int maxReps = 2;
//    int startCase = 1;
//    int endCase = 200000;

//    int startCase = 200001;
//    int endCase = 400000;
    //int startCase = 1;
    //int endCase = 4000000;

    string outputFileName = output_path+"/output.txt";
    string resultsFileName_success = output_path+"/output-results-success.txt";
    string resultsFileName_fail = output_path+"/output-results-fail.txt";

//    int startCase = 400001;
//    int endCase = 596499;
//    string outputFileName = "LuisOutC.txt";
    int testNum=getTestNum(input_file);

    TestFromFileCaseRange(input_file, outputFileName, resultsFileName_success, resultsFileName_fail, 0, testNum, 150, maxCopies, delPatternLen,
            subPriority, delPriority, insPriority, maxReps);

//    TestFromFile("evyaA.txt", testNum, 152, maxCopies, delPatternLen, subPriority, delPriority, insPriority,
//                maxReps);

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << endl;
    cout << "Time elapsed: " << (int) elapsed_secs << "\tseconds" << endl;
    return 0;
}

/*
int main() {
	clock_t begin = clock();



	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << endl;
	cout << "Time elapsed: " << (int) elapsed_secs << "\tseconds" << endl;
	return 0;
}*/

//	unsigned sd = chrono::high_resolution_clock::now().time_since_epoch().count();
//	int testNum = 1000;
//	int len = 50;
//	int cloneNum = 20;
//	double allProb = 0.10;
//	TestHeaviestPath(testNum, len, cloneNum, allProb, allProb, allProb, sd); // replaces Markov test
//	TestFixedLenMarkov(testNum, len, cloneNum, allProb, allProb, allProb, sd);
//	TestFixedLenMarkovOnCorrectedCluster(testNum, len, cloneNum, allProb, allProb, allProb, sd);
//	TestMinSumFixedLenMarkovCorrectedClusterTwice(testNum, len, cloneNum, allProb, allProb, allProb, sd);
//	TestMinSumFixedLenMarkovCorrectedClusterTwiceT(testNum, len, cloneNum, allProb, allProb, allProb, sd);
//	TestMinSumEDFixedLenMarkovMaxShiftCorrectedClusterTwice(testNum, len, cloneNum, allProb, allProb, allProb, sd,
//				maxDiagLongDim, maxDiagShortDim);
//	TestMinSumFixedLenMarkovCorrectedClusterTwiceJoinR(testNum, len, cloneNum, allProb, allProb, allProb, sd, R);
