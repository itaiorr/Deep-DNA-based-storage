
#ifndef EDITDISTANCE_HPP_
#define EDITDISTANCE_HPP_

#include <string>
#include <vector>
#include <random>
#include "Utils.hpp"
using namespace std;

//typedef void (*EditVectorFunc) (const string& X, const string& Y, mt19937& generator, EV& ev);


int EditDistance(const string& X, const string& Y);

void AnchorWAllPairsEV(const int anchorIndex, const vector<string>& Y, mt19937& generator, vector<EV>& allYEV);
void AnchorWAllPairsEV(const int anchorIndex, const vector<string>& Y, mt19937& generator, vector<EV>& allYEV,
		const int maxDiagLongDim, const int maxDiagShortDim);
void ConditionalFreqNVec(const int anchorIndex, const vector<string>& Y, mt19937& generator, int N,
		vector<CondFreq>& condFreqVec);

//map<string, double> CountOperations(const vector<LetterOps>& opList);


// diagonals of a rowsXcols matrix
// the diagonals: (0,maxDiag), (0,maxDiag-1),..., (0,0),..., (maxDiag,0)
class DiagMat {
	const int rows;
	const int cols;
	const int uppermostDiag; // left of diag (0,0). counted from (0,0)
	const int lowermostDiag; // right of diag (0,0). counted from (0,0)
	vector<int*> upperDiagsP;// pointer to first cell in diag for (0,0) diag and upper
	vector<int*> lowerDiagsP; // pointer to first cell in diag for (0,0) diag and lower
	vector<int> data; // all diagonals stored contiguously from uppermost to lowermost

public:
	DiagMat(const int rows, const int cols, const int uppermostDiag, const int lowermostDiag, const int initVal);
	// get reference to entry (i,j) as if it where a regular matrix
	int& operator()(const int i, const int j);
};

#endif /* EDITDISTANCE_HPP_ */
