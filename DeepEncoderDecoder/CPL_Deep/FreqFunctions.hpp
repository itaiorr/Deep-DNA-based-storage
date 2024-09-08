#ifndef FREQFUNCTIONS_HPP_
#define FREQFUNCTIONS_HPP_

#include <string>
#include <vector>
#include <random>
#include <map>
#include "EditDistance.hpp"
#include "Utils.hpp"

using namespace std;

const string START_STRING(1, END_SYMBOL);
const string END_STRING(1, END_SYMBOL);

void AddConditionalFreq(const EV& ev, vector<CondFreq>& condFreqVec);
void ProbVec(const int anchorIndex, const vector<string>& Y, mt19937& generator, vector<CondProb>& prob);
void ProbVec(const int anchorIndex, const vector<string>& Y, mt19937& generator, vector<CondProb>& prob,
		const int maxDiagLongDim, const int maxDiagShortDim);

#endif /* FREQFUNCTIONS_HPP_ */
