#include "Cluster.hpp"
#include <unordered_set>
#include <map>
#include <algorithm>

string MakeStrand(const unsigned length, mt19937& generator) {
	string strand;
	uniform_int_distribution<int> distribution(0, 3);
	string letters("GTAC");
	for (unsigned i = 0; i < length; i++) {
		strand += letters[distribution(generator)];
	}
	return strand;
}

string CopyStrand(const string& source, mt19937& generator, const double delProb, const double insProb,
		const double subProb) {
	uniform_real_distribution<double> distributionReal(0.0, 1.0);
	uniform_int_distribution<int> distributionInt(0, 3);
	uniform_int_distribution<int> distributionInt3(0, 2);
	string letters("GTAC");
	map<char, string> substitutions = { { 'A', "CGT" }, { 'C', "AGT" }, { 'G', "ACT" }, { 'T', "ACG" } };
	string copy;
	for (size_t i = 0; i < source.size(); i++) {
		if (distributionReal(generator) <= insProb) { // chose to insert
			copy += letters[distributionInt(generator)]; // choose random letter from GTAC
		}
		if (distributionReal(generator) > delProb) { // don't delete
			if (distributionReal(generator) <= subProb) { // substitute
				copy += substitutions[source[i]][distributionInt3(generator)];
			}
			else {	// just copy
				copy += source[i];
			}
		}
		else { // delete
			if (distributionReal(generator) <= subProb) { // delete + substitute = substitute
				copy += substitutions[source[i]][distributionInt3(generator)];
			}
		}
	}
	// insert at the end
	if (distributionReal(generator) <= insProb) {
		copy += letters[distributionInt(generator)];
	}
	return copy;
}

Cluster::Cluster(const unsigned strandLength, const int clonesNum, const double delProb, const double insProb,
		const double subProb, mt19937& generator) {

	// make original
	original = MakeStrand(strandLength, generator);

	unordered_set<string> tmpSet; // remember  ordered set can help choose a clone with less inserts
	// make clones
	while ((int) tmpSet.size() < clonesNum) {
		tmpSet.insert(CopyStrand(original, generator, delProb, insProb, subProb));
	}
	copies.insert(copies.end(), tmpSet.begin(), tmpSet.end());
	// shuffle to make order random in clones.
	shuffle(copies.begin(), copies.end(), generator);

}

// cluster of copies only of strandLength
Cluster::Cluster(const unsigned strandLength, const int clonesNum, const double delProb, const double insProb,
		const double subProb, mt19937& generator, bool dummy) {

	// make original
	original = MakeStrand(strandLength, generator);

	unordered_set<string> tmpSet; // remember  ordered set can help choose a clone with less inserts
	// make clones
	while ((int) tmpSet.size() < clonesNum) {
		string tempString = CopyStrand(original, generator, delProb, insProb, subProb);
		if (tempString.size() == strandLength and tempString != original)
			tmpSet.insert(tempString);
	}
	copies.insert(copies.end(), tmpSet.begin(), tmpSet.end());
	// shuffle to make order random in clones.
	shuffle(copies.begin(), copies.end(), generator);

}



Cluster::Cluster(const string& original, const vector<string>& copies):original(original),copies(copies){}

