#ifndef CLUSTER_HPP_
#define CLUSTER_HPP_

#include <string>
#include <vector>
#include <random>
using namespace std;

struct Cluster {
	string original;
	vector<string> copies;
	Cluster(){

	}
	Cluster(const unsigned strandLength, const int clonesNum, const double delProb, const double insProb,
			const double subProb, mt19937& generator);
	Cluster(const unsigned strandLength, const int clonesNum, const double delProb, const double insProb,
			const double subProb, mt19937& generator, bool dummy);
	Cluster(const string& original, const vector<string>& copies);
};

string MakeStrand(const unsigned length, mt19937& generator);

#endif /* CLUSTER_HPP_ */
