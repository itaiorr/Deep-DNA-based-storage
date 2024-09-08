
#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <limits>
#include <string>
#include <vector>
#include <map>

using namespace std;

const char END_SYMBOL='S';

const long double NINFLD = -numeric_limits<long double>::infinity();

typedef vector<string> EV; // Edit Vector
typedef map<string, pair<map<string, int>, int> > CondFreq;
typedef map<string, map<string, long double> > CondProb;
bool DefinitelyLessThan(long double a, long double b);
int SumED(const string& str, const vector<string>& strs);

#endif /* UTILS_HPP_ */
