#include <cfloat>
#include "Utils.hpp"
#include "EditDistance.hpp"

int SumED(const string& str, const vector<string>& strs) {
	int sumED = 0;
	for (auto& strng : strs) {
		sumED += EditDistance(str, strng);
	}
	return sumED;
}

bool DefinitelyLessThan(long double a, long double b) {
	return (b - a) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * LDBL_EPSILON);
}
