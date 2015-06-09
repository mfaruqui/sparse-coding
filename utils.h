#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <functional>
#include <cmath>
#include <ctype.h>
#include <string>
#include <tr1/unordered_map>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

typedef Matrix<double, Dynamic, 1> Col;
typedef Matrix<double, 1, Dynamic> Row;
typedef Matrix<double, Dynamic, Dynamic> Mat;

typedef std::tr1::unordered_map<int, Col> mapIntCol;
typedef std::tr1::unordered_map<int, Mat> mapIntMat;
typedef std::tr1::unordered_map<int, unsigned> mapIntUnsigned;
typedef std::tr1::unordered_map<unsigned, unsigned> mapUnsUns;
typedef std::tr1::unordered_map<unsigned, double> mapUnsDouble;

typedef vector<string> mapUnsignedStr;

vector<string> split_line(const string&, char);

void ReadVecsFromFile(const string&, mapUnsignedStr&, vector<Col>&);

void ElemwiseTanh(Col*);
void ElemwiseTanhGrad(const Col&, Col*);

void ElemwiseAndrewNsnl(Col*);
void ElemwiseAndrewNsnlGrad(const Col&, Col*);

#endif
