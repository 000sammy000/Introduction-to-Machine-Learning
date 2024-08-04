#include "helper.h"
using namespace std;
using PVV = pair<vector<int>, vector<int>>;

vector<vector<float>> read_csv_file(string &file_pth) {
    ifstream dataFile(file_pth);
    if (!dataFile.is_open()) {
        cerr << "Error opening file: " << file_pth << endl;
        return {};
    }

    vector<vector<float>> data;
    cout<<file_pth<<" open\n";
    string line;
    bool skip_first_row = true;
    while (getline(dataFile, line)) {
        if (skip_first_row) {
            skip_first_row = false;
            continue;
        }
        vector<float> row;
        if(line[line.size()-1]==','){
            line+="-0.1";
        }
        stringstream ss(line);
        string cell;
        
            
        while (getline(ss, cell, ',')) {
            
            if (!cell.empty()) {
                row.push_back(stof(cell));
            } else {
                row.push_back(-1.0); 
            }
        }
        
        data.push_back(row);
    }
    return data;
}

vector<vector<float>> get_X(vector<vector<float>> &df) {
    vector<vector<float>> X;
    for (auto &row: df) {
        vector<float> tmp_row = row;
        tmp_row.pop_back();
        X.push_back(tmp_row);
    }
    return X;
}
vector<vector<float>> get_X_test(vector<vector<float>> &df) {
    vector<vector<float>> X;
    for (auto &row: df) {
        X.push_back(row);
    }
    return X;
}

vector<vector<float>> get_y(vector<vector<float>> &df) {
    vector<vector<float>> y;
    for (auto &row: df) {
        y.push_back({row.back()});
    }
    return y;
}

vector<PVV> k_fold_split(vector<vector<float>>& X_train, vector<vector<float>>& y_train, int n_splits, int random_state) {
    int data_size = X_train.size();
    vector<int> indices(data_size);
    iota(indices.begin(), indices.end(), 0); 
    /*shuffle dataset*/
    mt19937 rg(random_state);
    shuffle(indices.begin(), indices.end(), rg);

    vector<PVV> folds;
    int fold_size = data_size / n_splits, remainder = data_size % n_splits;
    int val_start = 0;

    for (int i = 0; i < n_splits; i++) {
        int fold_additional = (i < remainder) ? 1: 0;
        int val_end = val_start + fold_size + fold_additional;

        vector<int> val_indices(indices.begin() + val_start, indices.begin() + val_end);
        vector<int> train_indices;
        train_indices.reserve(data_size - val_indices.size());  // initial vector size
        train_indices.insert(train_indices.end(), indices.begin(), indices.begin() + val_start);
        train_indices.insert(train_indices.end(), indices.begin() + val_end, indices.end());

        folds.push_back(make_pair(train_indices, val_indices));
        val_start = val_end;

    }
    return folds;
}

float accuracy_score(vector<vector<float>>& y_test, vector<vector<float>>& predictions) {
    int num_samples = y_test.size();
    float correct_predictions = 0.0;
    for (int i = 0; i < num_samples; i++) {
        //if(i<5)cout<<y_test[i][0]<<" "<<predictions[i][0]<<endl;
        if (y_test[i][0] == predictions[i][0]) {
            correct_predictions += 1.0f;
        }
    }
    return correct_predictions / num_samples;
}

float precision_score(vector<vector<float>>& y_test, vector<vector<float>>& predictions) {
    int num_samples = y_test.size();
    int tp = 0, fp = 0;

    for (int i = 0; i < num_samples; i++) {
        if (y_test[i][0] == 1.0f && predictions[i][0] == 1.0f) {
            tp++;
        }
        else if (y_test[i][0] == 0.0f && predictions[i][0] == 1.0f) {
            fp++;
        }
    }
    float precision = 0.0f;
    if (tp + fp != 0) {
        precision = static_cast<float>(tp) / (tp + fp);
    }
    return precision;
}

float recall_score(vector<vector<float>>& y_test, vector<vector<float>>& predictions) {
    int num_samples = y_test.size();
    int tp = 0, fn = 0;
    
    for (int i = 0; i < num_samples; i++) {

        if (y_test[i][0] == 1.0f && predictions[i][0] == 1.0f) {
            tp++;
        }
        else if (y_test[i][0] == 1.0f && predictions[i][0] == 0.0f) {
            fn++;
        }
    }
    float recall = 0.0f;
    if (tp + fn != 0) {
        recall = static_cast<float>(tp) / (tp + fn);
    }
    // cout << "tp = " << tp <<  " fn = "<< fn << endl;
    return recall;
}

float f1_score(float recall, float precision) {
    float f1 = 2 * (precision * recall) / (precision + recall);
    return f1;
}

float matthews_corrcoef(vector<vector<float>>& y_test, vector<vector<float>>& predictions) {
    int num_samples = y_test.size();
    float tp = 0.0f, tn = 0.0f, fp = 0.0f, fn = 0.0f;

    for (int i = 0; i < num_samples; i++) {
        if (y_test[i][0] == 1.0f && predictions[i][0] == 1.0f) {
            tp++;
        }
        else if (y_test[i][0] == 0.0f && predictions[i][0] == 1.0f) {
            fp++;
        }
        else if (y_test[i][0] == 1.0f && predictions[i][0] == 0.0f) {
            fn++;
        }
        else {
            tn++;
        }
    }
    float mcc = 0.0f;
    float denominator = 0.0f, numerator = 0.0f;
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    numerator = (tp * tn) - (fp * fn);
    if (denominator != 0 ) {
        mcc = numerator / denominator;
    }
    return mcc;
}

double AUROC(vector<float>& labels, vector<float>& scores, int n) {
	for (int i = 0; i < n; i++)
		if (!isfinite(scores[i]) || labels[i] != 0 && labels[i] != 1)
			return numeric_limits<double>::signaling_NaN();

	const auto order = new int[n];
	iota(order, order + n, 0);
	sort(order, order + n, [&](int a, int b) { return scores[a] > scores[b]; });
	const auto y = new double[n];
	const auto z = new double[n];
	for (int i = 0; i < n; i++) {
		y[i] = labels[order[i]];
		z[i] = scores[order[i]];
	}

	const auto tp = y; // Reuse
	partial_sum(y, y + n, tp);

	int top = 0; // # diff
	for (int i = 0; i < n - 1; i++)
		if (z[i] != z[i + 1])
			order[top++] = i;
	order[top++] = n - 1;
	n = top; // Size of y/z -> sizeof tps/fps

	const auto fp = z; // Reuse
	for (int i = 0; i < n; i++) {
		tp[i] = tp[order[i]]; // order is mono. inc.
		fp[i] = 1 + order[i] - tp[i]; // Type conversion prevents vectorization
	}
	delete[] order;

	const auto tpn = tp[n - 1], fpn = fp[n - 1];
	for (int i = 0; i < n; i++) { // Vectorization
		tp[i] /= tpn;
		fp[i] /= fpn;
	}

	auto area = tp[0] * fp[0] / 2; // The first triangle from origin;
	double partial = 0; // For Kahan summation
	for (int i = 1; i < n; i++) {
		const auto x = (fp[i] - fp[i - 1]) * (tp[i] + tp[i - 1]) / 2 - partial;
		const auto sum = area + x;
		partial = (sum - area) - x;
		area = sum;
	}

	delete[] tp;
	delete[] fp;

	return area;
}

double roc_auc_score(vector<vector<float>>& y_test, vector<unordered_map<int, float>>& proba_array) {
    int num_samples = y_test.size();
    vector<float> labels(num_samples);
    vector<float> scores(num_samples);

    for (int i=0; i < num_samples; i++) {
        labels[i] = y_test[i][0];
        scores[i] = proba_array[i][1];
    }
    double auc = AUROC(labels, scores, num_samples);
    return auc;
}

void write_result_to_csv(vector<vector<unordered_map<string, float>>>& cv_result) {
    ofstream file("cv_result.csv");
    if (file.is_open()) {
        file << "model,accuracy,f1,precision,recall,mcc,auc\n";
        for (auto &fold_result: cv_result) {
            for (auto &res: fold_result) {
                file << res.at("model") << ','
                    << res.at("accuracy") << ','
                    << res.at("f1") << ','
                    << res.at("precision") << ','
                    << res.at("recall") << ','
                    << res.at("mcc") << ','
                    << res.at("auc") << '\n';
            }
        }
        file.close();
        cout << "write csv result is complete\n";
    }
    else {cerr << "can't open the file\n";}
}

unordered_map<float, vector<vector<float>>> predictions = {{1.0f, {{1}, {0}, {0}}},
                                                            {2.0f, {{1}, {1}, {0}}},
                                                            {3.0f, {{0}, {0}, {0}}}};

void write_predictions_to_csv(unordered_map<float, vector<vector<float>>>& predictions) {
    ofstream file("test_results.csv");
    if (file.is_open()) {
        file << "model,predictions\n";
        for (auto& entry : predictions) {
            float model = entry.first;
            auto& pred = entry.second;

            for (auto& p : pred) {
                file << model << ',' << p[0] << '\n';
            }
        }

        file.close();
    }
    else {cerr << "can't open the testing file\n";}
}