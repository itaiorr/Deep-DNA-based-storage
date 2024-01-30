#include <cassert>
#include <climits>
#include <algorithm>
#include "EditDistance.hpp"
#include "FreqFunctions.hpp"

int DiagLen(const int mRows, const int mCols, const int startRow, const int startCol) {
	return min(mRows - startRow, mCols - startCol);
}

DiagMat::DiagMat(const int rows, const int cols, const int uppermostDiag, const int lowermostDiag, const int initVal) :
		rows(rows), cols(cols), uppermostDiag(uppermostDiag), lowermostDiag(lowermostDiag), upperDiagsP(
				uppermostDiag + 1), lowerDiagsP(lowermostDiag + 1) {
	assert(lowermostDiag > 0 && lowermostDiag < rows && uppermostDiag > 0 && uppermostDiag < cols);
	int dataSize;
	if (cols >= rows) {
		int upperDataSize, lowerDataSize;
		if (uppermostDiag > cols - rows) {
			// data above (0,n-m) diag
			int upperTrapezSize = (cols + rows - uppermostDiag - 1) * (uppermostDiag - (cols - rows)) / 2;

			// data from diag (0,0) to diag (0, cols-rows)
			int parallelSize = (cols - rows + 1) * rows;

			upperDataSize = upperTrapezSize + parallelSize;
		}
		else { // uppermostDiag <= cols-rows
			   // data from diag (0,0) to diag (0,uppermostDiag)
			int parallelSize = (uppermostDiag + 1) * rows;
			upperDataSize = parallelSize;
		}
		// data from diag(1,0) to diag (lowermostDiag,0)
		int lowerTrapezSize = (2 * rows - lowermostDiag - 1) * lowermostDiag / 2;
		lowerDataSize = lowerTrapezSize;
		dataSize = upperDataSize + lowerDataSize;
	}
	else {			   // cols < rows
		int upperDataSize, lowerDataSize;
		if (lowermostDiag > rows - cols) {
			// data below (m-n,0) diag
			int lowerTrapezSize = (cols + rows - lowermostDiag - 1) * (lowermostDiag - (rows - cols)) / 2;

			// data from diag (0,0) to diag (lowermostDiag,0)
			int parallelSize = (rows - cols + 1) * cols;

			lowerDataSize = lowerTrapezSize + parallelSize;
		}
		else { // lowermostDiag <= rows-cols
			   // data from diag (0,0) to diag (lowermostDiag,0)
			int parallelSize = (lowermostDiag + 1) * cols;
			lowerDataSize = parallelSize;
		}
		int upperTrapezSize = (2 * cols - uppermostDiag - 1) * uppermostDiag / 2;
		upperDataSize = upperTrapezSize;
		dataSize = upperDataSize + lowerDataSize;
	}
	data = vector<int>(dataSize, initVal);

	// pointers to first cell in each diag
	// upper diags pointers
	int* dataPointer = &data[0];
	int totalDataSize = 0; // for debugging only
	for (int upperDiag = uppermostDiag; upperDiag >= 1; upperDiag--) {
		upperDiagsP[upperDiag] = dataPointer;
		dataPointer += DiagLen(rows, cols, 0, upperDiag);
		totalDataSize += DiagLen(rows, cols, 0, upperDiag);
	}
	// diag (0,0);
	upperDiagsP[0] = dataPointer;
	lowerDiagsP[0] = dataPointer;
	dataPointer += DiagLen(rows, cols, 0, 0);
	totalDataSize += DiagLen(rows, cols, 0, 0);

	// lower diags pointers
	for (int lowerDiag = 1; lowerDiag <= lowermostDiag; lowerDiag++) {
		lowerDiagsP[lowerDiag] = dataPointer;
		dataPointer += DiagLen(rows, cols, lowerDiag, 0);
		totalDataSize += DiagLen(rows, cols, lowerDiag, 0);
	}
	assert(totalDataSize == dataSize);
}

int& DiagMat::operator()(const int i, const int j) {
	assert(i >= 0 && i < rows);
	assert(j >= 0 && j < cols);
	if (i > j) {
		assert(i - j <= lowermostDiag);
		return lowerDiagsP[i - j][j];
	}
	else { // i <= j
		assert(j - i <= uppermostDiag);
		return upperDiagsP[j - i][i];
	}
}

int EditDistanceArray(const string& X, const string& Y, int m, int n, vector<vector<int> >& dp) {
	for (int i = 0; i <= m; i++) {
		for (int j = 0; j <= n; j++) {
			if (i == 0) {
				dp[i][j] = j;
			}
			else if (j == 0) {
				dp[i][j] = i;
			}
			else if (X[i - 1] == Y[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1];
			}
			else {
				dp[i][j] = 1 + min(min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]);
			}
		}
	}
	return dp[m][n];
}

int EditDistanceArray(const string& X, const string& Y, int m, int n, int uppermostDiag, int lowermostDiag,
		DiagMat& dp) {
	for (int i = 0; i <= m; i++) {
		for (int j = 0; j <= n; j++) {
			if (i - j > lowermostDiag || j - i > uppermostDiag)
				continue;
			if (i == 0) {
				dp(i, j) = j;
			}
			else if (j == 0) {
				dp(i, j) = i;
			}
			else if (X[i - 1] == Y[j - 1]) {
				dp(i, j) = dp(i - 1, j - 1);
			}
			else {
				dp(i, j) = 1 + min(min(dp(i, j - 1), dp(i - 1, j)), dp(i - 1, j - 1));
			}
		}
	}
	return dp(m, n);
}

// m<=n
int EditDistanceArraymLTEn(const string& X, const string& Y, int m, int n, int uppermostDiag, int lowermostDiag,
		DiagMat& dp) {
	assert(m <= n);
	for (int i = 0; i <= m; i++) {
		for (int j = max(0, i - lowermostDiag); j <= min(uppermostDiag + i, n); j++) {
			if (i == 0) {
				dp(i, j) = j;
			}
			else if (j == 0) {
				dp(i, j) = i;
			}
			else if (X[i - 1] == Y[j - 1]) {
				dp(i, j) = dp(i - 1, j - 1);
			}
			else {
				dp(i, j) = 1 + min(min(dp(i, j - 1), dp(i - 1, j)), dp(i - 1, j - 1));
			}
		}
	}
	return dp(m, n);
}

// m>n
int EditDistanceArraymGTn(const string& X, const string& Y, int m, int n, int uppermostDiag, int lowermostDiag,
		DiagMat& dp) {
	assert(m > n);
	for (int j = 0; j <= n; j++) {
		for (int i = max(0, j - uppermostDiag); i <= min(lowermostDiag + j, m); i++) {
			if (i == 0) {
				dp(i, j) = j;
			}
			else if (j == 0) {
				dp(i, j) = i;
			}
			else if (X[i - 1] == Y[j - 1]) {
				dp(i, j) = dp(i - 1, j - 1);
			}
			else {
				dp(i, j) = 1 + min(min(dp(i, j - 1), dp(i - 1, j)), dp(i - 1, j - 1));
			}
		}
	}
	return dp(m, n);
}

int EditDistance(const string& X, const string& Y) {

	int m = X.length(), n = Y.length();
	vector<vector<int> > dp(m + 1, vector<int>(n + 1));

	return EditDistanceArray(X, Y, m, n, dp);
}

// maxShift - max deletions without insert between them or Insertions without deletion between them
int EditDistance(const string& X, const string& Y, const int maxDiagLongDim, const int maxDiagShortDim) {
	int m = X.length(), n = Y.length();
	int lowermostDiag, uppermostDiag;
	if (m <= n) { // n is long dim
		assert(maxDiagLongDim >= n - m);
		uppermostDiag = maxDiagLongDim;
		lowermostDiag = maxDiagShortDim;
	}
	else { //m>n. n is short dim
		assert(maxDiagLongDim >= m - n);
		uppermostDiag = maxDiagShortDim;
		lowermostDiag = maxDiagLongDim;
	}
	assert(uppermostDiag < n);
	assert(lowermostDiag < m);
	DiagMat dp(m + 1, n + 1, uppermostDiag + 1, lowermostDiag + 1, INT_MAX); // with additional diag on both sides to enable computation of maxDiag

	return EditDistanceArray(X, Y, m, n, uppermostDiag, lowermostDiag, dp);
}

void BacktrackEditVector(const string& X, const string& Y, const vector<vector<int> >& dp, mt19937& generator, EV& s) {
	int m = X.size();
	int n = Y.size();
	s = EV(2 * m + 1);

	while (1) {
		if (m == 0) {
			s[0] = Y.substr(0, n);
			break;
		}
		else if (n == 0) {
			for (int index = 0; index < m; index++) {
				s[2 * index + 1] = "";
			}
			break;
		}

		if (X[m - 1] == Y[n - 1]) {
			s[2 * m - 1] = string(1, X[m - 1]);
			m--;
			n--;
		}
		else {
			vector<int> nextStepCodes;
			if (dp[m][n] == dp[m - 1][n - 1] + 1) {  	// replace
				nextStepCodes.push_back(0);
			}
			if (dp[m][n] == dp[m][n - 1] + 1) { 		// insert
				nextStepCodes.push_back(1);
			}
			if (dp[m][n] == dp[m - 1][n] + 1) {			// delete
				nextStepCodes.push_back(2);
			}
			if (nextStepCodes.size() > 1) {
				shuffle(nextStepCodes.begin(), nextStepCodes.end(), generator);
			}
			int stepCode = nextStepCodes[0];
			switch (stepCode) {
			case 0: // replace
				s[2 * m - 1] = string(1, Y[n - 1]);
				m--;
				n--;
				break;
			case 1: // insert
				s[2 * m].insert(s[2 * m].begin(), Y[n - 1]);
				n--;
				break;
			case 2: // delete
				s[2 * m - 1] = "";
				m--;
				break;
			}
		}
	}
	return;
}

void BacktrackEditVector(const string& X, const string& Y, DiagMat& dp, mt19937& generator, EV& s) {
	int m = X.size();
	int n = Y.size();
	s = EV(2 * m + 1);

	while (1) {
		if (m == 0) {
			s[0] = Y.substr(0, n);
			break;
		}
		else if (n == 0) {
			for (int index = 0; index < m; index++) {
				s[2 * index + 1] = "";
			}
			break;
		}

		if (X[m - 1] == Y[n - 1]) {
			s[2 * m - 1] = string(1, X[m - 1]);
			m--;
			n--;
		}
		else {
			vector<int> nextStepCodes;
			if (dp(m, n) == dp(m - 1, n - 1) + 1) {  	// replace
				nextStepCodes.push_back(0);
			}
			if (dp(m, n) == dp(m, n - 1) + 1) { 		// insert
				nextStepCodes.push_back(1);
			}
			if (dp(m, n) == dp(m - 1, n) + 1) {			// delete
				nextStepCodes.push_back(2);
			}
			if (nextStepCodes.size() > 1) {
				shuffle(nextStepCodes.begin(), nextStepCodes.end(), generator);
			}
			int stepCode = nextStepCodes[0];
			switch (stepCode) {
			case 0: // replace
				s[2 * m - 1] = string(1, Y[n - 1]);
				m--;
				n--;
				break;
			case 1: // insert
				s[2 * m].insert(s[2 * m].begin(), Y[n - 1]);
				n--;
				break;
			case 2: // delete
				s[2 * m - 1] = "";
				m--;
				break;
			}
		}
	}
	return;
}

void EditVector(const string& X, const string& Y, mt19937& generator, EV& ev) {

	int m = X.length(), n = Y.length();
	vector<vector<int> > dp(m + 1, vector<int>(n + 1));
	EditDistanceArray(X, Y, m, n, dp);
	BacktrackEditVector(X, Y, dp, generator, ev);
	return;
}

void EditVector(const string& X, const string& Y, const int maxDiagLongDim, const int maxDiagShortDim,
		mt19937& generator, EV& ev) {

	int m = X.length(), n = Y.length();
	int lowermostDiag, uppermostDiag;
	if (m <= n) { // n is long dim
		uppermostDiag = maxDiagLongDim;
		lowermostDiag = maxDiagShortDim;
		if (uppermostDiag < n - m)
			uppermostDiag = n - m;
	}
	else { //m>n. n is short dim
		uppermostDiag = maxDiagShortDim;
		lowermostDiag = maxDiagLongDim;
		if (lowermostDiag < m - n)
			lowermostDiag = m - n;
	}
	assert(uppermostDiag < n);
	assert(lowermostDiag < m);
	DiagMat dp(m + 1, n + 1, uppermostDiag + 1, lowermostDiag + 1, INT_MAX); // with additional diag on both sides to enable computation of maxDiag
	if (m <= n)
		EditDistanceArraymLTEn(X, Y, m, n, uppermostDiag, lowermostDiag, dp);
	else
		EditDistanceArraymGTn(X, Y, m, n, uppermostDiag, lowermostDiag, dp);
	BacktrackEditVector(X, Y, dp, generator, ev);
	return;
}

void AnchorWAllPairsEV(const int anchorIndex, const vector<string>& Y, mt19937& generator, vector<EV>& allYEV) {
	int copyNum = Y.size();
	allYEV = vector<EV>(copyNum - 1);
	int pairIndex = 0;
	for (int i = 0; i < copyNum; i++) {
		if (i == anchorIndex)
			continue;
		EditVector(Y[anchorIndex], Y[i], generator, allYEV[pairIndex++]);
	}
}


void AnchorWAllPairsEV(const int anchorIndex, const vector<string>& Y, mt19937& generator, vector<EV>& allYEV,
		const int maxDiagLongDim, const int maxDiagShortDim) {
	int copyNum = Y.size();
	allYEV = vector<EV>(copyNum - 1);
	int pairIndex = 0;
	for (int i = 0; i < copyNum; i++) {
		if (i == anchorIndex)
			continue;
		EditVector(Y[anchorIndex], Y[i], maxDiagLongDim, maxDiagShortDim, generator, allYEV[pairIndex++]);
	}
}


int ComputeEditDistanceNum(const string& X, const string& Y) {

    int m = X.length(), n = Y.length();
    vector<vector<int> > dp(m + 1, vector<int>(n + 1));

    return EditDistanceArray(X, Y, m, n, dp);
}
