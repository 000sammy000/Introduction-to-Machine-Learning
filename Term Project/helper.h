#ifndef _HELPER_H_
#define _HELPER_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include <unordered_map>
#include <string>
#include <random>
// #include <numeric>
#include <numeric>
#include <algorithm>


using namespace std;
using PVV = pair<vector<int>, vector<int>>;

vector<vector<float>> read_csv_file(string&);
vector<vector<float>> get_X(vector<vector<float>>&);
vector<vector<float>> get_X_test(vector<vector<float>>&);
vector<vector<float>> get_y(vector<vector<float>>&);
vector<PVV> k_fold_split(vector<vector<float>>&, vector<vector<float>>&, int, int);
float accuracy_score(vector<vector<float>>&, vector<vector<float>>&);
float recall_score(vector<vector<float>>&, vector<vector<float>>&);
float precision_score(vector<vector<float>>&, vector<vector<float>>&);
float f1_score(float, float);
float matthews_corrcoef(vector<vector<float>>&, vector<vector<float>>&);
double roc_auc_score(vector<vector<float>>&, vector<unordered_map<int, float>>&);
double AUROC(vector<float>&, vector<float>&, int);
void write_result_to_csv(vector<vector<unordered_map<string, float>>>&);
void write_predictions_to_csv(unordered_map<float, vector<vector<float>>>&);

#endif