#include <cassert>
#include "FreqFunctions.hpp"
#include "Graph.hpp"

// Add EV to CondFreq vector
void AddConditionalFreq(const EV& ev, vector<CondFreq>& condFreqVec) {
	int evSize = ev.size();
	for (int i = 0; i < evSize - 1; i++) {
		condFreqVec[i][ev[i]].first[ev[i + 1]]++; // increase count of ev[i+1] given ev[i] now
		condFreqVec[i][ev[i]].second++; // increase count of apperances of string ev[i]
	}
	// last string in ev
	condFreqVec[evSize - 1][ev[evSize - 1]].first[END_STRING]++; // only END_STRING can be next
	condFreqVec[evSize - 1][ev[evSize - 1]].second++;
}

void ConditionalFreqVec(const vector<EV>& vecEV, vector<CondFreq>& condFreqVec) {
	assert(not vecEV.empty());
	int evSize = vecEV[0].size();
	condFreqVec = vector<CondFreq>(evSize);
	for (auto& ev : vecEV) {
		AddConditionalFreq(ev, condFreqVec);
	}
}

// Count total frequencies of strings in main map
long double TotalFreq(const CondFreq& condFreq) {
	assert(not condFreq.empty());
	long double totalFreq = 0;
	for (auto& strPairPr : condFreq) {
		totalFreq += strPairPr.second.second;
	}
	return totalFreq;
}

// Convert vector of CondFreq to vector of CondProb.
void FreqToProbLog(const vector<CondFreq>& freqVec, vector<CondProb>& probVec) {

	assert(not freqVec.empty());
	long double totalFreq = TotalFreq(freqVec[0]);
//	assert(TotalFreq(freqVec[10]) == totalFreq);

	probVec = vector<CondProb>(freqVec.size());
	//multiply first conditional probabilities by string probability because: prob (path ABC)= prob(A)*prob(B|A)*prob(B|C)
	for (auto& stringPairPr : freqVec[0]) {
		const string& fromStr = stringPairPr.first;
//		long double fromStrFreq = stringPairPr.second.second;
		for (auto& pr : stringPairPr.second.first) {
			const string& toStr = pr.first;
			long double toStrFreq = pr.second;
			probVec[0][fromStr][toStr] = log(toStrFreq / totalFreq); // (toStrFreq / fromStrFreq)*(fromStrFreq/totalFreq)=toStrFreq/totalFreq
		}
	}

	for (int i = 1; i < (int) freqVec.size(); i++) {
		for (auto& stringPairPr : freqVec[i]) {
			const string& fromStr = stringPairPr.first;
			long double fromStrFreq = stringPairPr.second.second;
			for (auto& pr : stringPairPr.second.first) {
				const string& toStr = pr.first;
				long double toStrFreq = pr.second;
				probVec[i][fromStr][toStr] = log(toStrFreq / fromStrFreq);
			}
		}
	}
}

void ProbVec(const int anchorIndex, const vector<string>& Y, mt19937& generator, vector<CondProb>& prob) {
	vector<CondFreq> freq;
	vector<EV> allYEV;
	AnchorWAllPairsEV(anchorIndex, Y, generator, allYEV);
	ConditionalFreqVec(allYEV, freq);
	FreqToProbLog(freq, prob);
}

void ProbVec(const int anchorIndex, const vector<string>& Y, mt19937& generator, vector<CondProb>& prob,
		const int maxDiagLongDim, const int maxDiagShortDim) {
	vector<CondFreq> freq;
	vector<EV> allYEV;
	AnchorWAllPairsEV(anchorIndex, Y, generator, allYEV, maxDiagLongDim, maxDiagShortDim);
	ConditionalFreqVec(allYEV, freq);
	FreqToProbLog(freq, prob);
}
