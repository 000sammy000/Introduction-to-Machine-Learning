#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "helper.h"

using namespace std;
using PVV = pair<vector<int>, vector<int>>;

vector<vector<float>>preprocess_parameter(77,vector<float>(2));
int IsVal;

class Preprocessor {
public:
    Preprocessor(vector<vector<float>> &data) : df(data) {}

    vector<vector<float>> preprocess() {
        _preprocess_categorical();
        _preprocess_numerical();
        _preprocess_ordinal();
        return df;
    }
private:
    vector<vector<float>> df;

    void _preprocess_categorical() {   
        for (size_t col = 17; col < 77; ++col) {
            
            if(IsVal==0){
                int cnt0=0,cnt1=0;
                for (int row = 0; row < df.size(); ++row) {
                    if(df[row][col]==0)cnt0++;
                    if(df[row][col]==1)cnt1++;
                }
                int mod;
                if(cnt0>cnt1){
                    mod=0;
                }else{
                    mod=1;
                }
                preprocess_parameter[col][0]=mod;
            }

            for(int row=0;row<df.size();row++){
                if(df[row][col]==-1)df[row][col]=preprocess_parameter[col][0];
            }
            
        }
    }

    void _preprocess_numerical() {
        
        for (int col = 0; col < 17; ++col) {
            if(IsVal==0){
                float mean_val = 0.0;
                for (int row = 0; row < df.size(); ++row) {
                    if(df[row][col]!=-1)mean_val += df[row][col];
                }
                mean_val /= df.size();
                preprocess_parameter[col][0]=mean_val;

                float std_dev = 0.0;
                for (int row = 0; row < df.size(); ++row) {
                    if(df[row][col]!=-1)std_dev += pow(df[row][col] - mean_val, 2);
                }
                std_dev = sqrt(std_dev / df.size());
                preprocess_parameter[col][1]=std_dev;
            }

            // Apply Z-score normalization to each element in the column
            for (int row = 0; row < df.size(); ++row) {
                if(df[row][col]!=-1)df[row][col] = (df[row][col] - preprocess_parameter[col][0]) / preprocess_parameter[col][1];
                if(df[row][col]==-1)df[row][col]=0;
            }
        }
        
    }

    void _preprocess_ordinal() {
        // Custom logic for preprocessing ordinal features goes here
    }
};


class Classifier {
public:
    virtual void fit(vector<vector<float>> &X, vector<vector<float>> &y) = 0;
    virtual vector<vector<float>> predict(vector<vector<float>> &X) = 0;
    virtual vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) = 0;
};

class NaiveBayesClassifier: public Classifier {
public:
    void fit(vector<vector<float>> &X, vector<vector<float>> &y) override {
        
        unordered_map<int, vector<vector<float>>> separated_data;
        for (size_t i = 0; i < X.size(); ++i) {
            int label = y[i][0];
            separated_data[label].push_back(X[i]);
        }

        size_t total_samples = X.size();
        for (auto &entry : separated_data) {
            int label = entry.first;
            size_t class_samples = entry.second.size();
            float class_probability = static_cast<float>(class_samples) / total_samples;
            class_probabilities[label] = {class_probability};
        }


        for (auto &entry : separated_data) {
            int label = entry.first;
            vector<vector<float>> class_data = entry.second;
            size_t num_features = class_data[0].size();

            vector<float> mean_values(17, 0.0);
            vector<float> std_dev_values(17, 0.0);

            for (size_t i = 0; i < 17; ++i) {
                for (size_t j = 0; j < class_data.size(); ++j) {
                    mean_values[i] += class_data[j][i];
                }
                mean_values[i] /= class_data.size();
            }

            for (size_t i = 0; i < 17; ++i) {
                for (size_t j = 0; j < class_data.size(); ++j) {
                    std_dev_values[i] += pow(class_data[j][i] - mean_values[i], 2);
                }
                std_dev_values[i] = sqrt(std_dev_values[i] / class_data.size());
            }

            vector<vector<float>> parameters;
            parameters.push_back(mean_values);
            parameters.push_back(std_dev_values);
            class_parameters[label] = parameters;

            
            for(size_t i=17;i<77;i++){
                float class_feature_count[2]={0,0};
                for (size_t j = 0; j < class_data.size(); ++j) {
                    if(class_data[j][i]==1){
                        class_feature_count[1]++;
                    }else{
                        class_feature_count[0]++;
                    }
                }
                
                class_parameters[label][0].push_back(class_feature_count[0]/(float)class_data.size());
                class_parameters[label][1].push_back(class_feature_count[1]/(float)class_data.size());

            }
        
        }
        
    }
    vector<vector<float>> predict(vector<vector<float>> &X) override {
        // Implement the prediction logic for Naive Bayes classifier
        vector<vector<float>> predictions;

        float zero=0.00001;
        for (auto &sample : X) {
            unordered_map<int,float> result;

            for (auto &entry : class_probabilities) {
                int label = entry.first;
                vector<float> class_prob = entry.second;

                float log_probability = log(class_prob[0]);

                for (size_t i = 0; i < 17; ++i) {
                    float mean = class_parameters[label][0][i];
                    float std_dev = class_parameters[label][1][i];

                    float exponent = exp(-pow(sample[i] - mean, 2) / (2 * pow(std_dev, 2)));
                    float feature_probability = (1 / (sqrt(2 * 3.14) * std_dev)) * exponent;
                    if(feature_probability==0){
                        feature_probability=zero/(float)class_prob[1];
                    }
    
                    log_probability += log(feature_probability);
                }
                for(size_t i=17;i<77;i++){
                    float feature_probability=class_parameters[label][sample[i]][i];
                    if(feature_probability==0){
                        feature_probability=zero;
                    }
                    log_probability+=log(feature_probability);
                }
                result[label]=log_probability;
            }
            
            int predicted_label;
            if(result[1]>result[0]){
                predicted_label=1;
            }else{
                predicted_label=0;
            }
            predictions.push_back({static_cast<float>(predicted_label)});
        }

        return predictions;
        
    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) {
        // Implement probability estimation for Naive Bayes classifier
        vector<unordered_map<int, float>> probabilities;
        float zero=0.00001;
        for (auto &sample : X) {
            unordered_map<int, float> class_probability;
            float total_prob=0;
            for (auto &entry : class_probabilities) {
                int label = entry.first;
                vector<float> class_prob = entry.second;

                float final_probability = log(class_prob[0]);

                for (size_t i = 0; i < 17; ++i) {
                    float mean = class_parameters[label][0][i];
                    float std_dev = class_parameters[label][1][i];

                    float exponent = exp(-pow(sample[i] - mean, 2) / (2 * pow(std_dev, 2)));
                    float feature_probability = (1 / (sqrt(2 * 3.14) * std_dev)) * exponent;
                    if(feature_probability==0){
                        feature_probability=zero;
                    }
                    final_probability += log(feature_probability);
                }

                for(size_t i=17;i<77;i++){
                    float feature_probability=class_parameters[label][sample[i]][i];
                    if(feature_probability==0){
                        feature_probability=zero;
                    }
                    final_probability+=log(feature_probability);
                }
                total_prob+=exp(final_probability);
                
                class_probability[label] = exp(final_probability);
            }
            class_probability[0]=class_probability[0]/total_prob;
            class_probability[1]=class_probability[1]/total_prob;
            
            probabilities.push_back(class_probability);
        }

        return probabilities;
    }

private: 
    // Implement private function or variable if you needed
    unordered_map<int, vector<float>> class_probabilities;
    unordered_map<int, vector<vector<float>>> class_parameters;

};

class KNearestNeighbors: public Classifier {
public:
    KNearestNeighbors(int k = 3): k(k) {
        
    } 

    void fit(vector<vector<float>> &X, vector<vector<float>> &y) override {
        // Implement the fitting logic for KNN
        train_X=X;
        train_y=y;
    }
    vector<vector<float>> predict(vector<vector<float>> &X) override {
        // Implement the prediction logic for KNN
        vector<vector<float>> predictions;
        for(const auto &sample:X){
            vector<float>distances;
            for(int j=0;j<train_X.size();j++){
                float dis=coutDistance(sample,train_X[j]);
                distances.push_back(dis);
                
            }
            vector<int> indices(distances.size());
            iota(indices.begin(), indices.end(), 0);
            partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                [&distances](int i, int j) { return distances[i] < distances[j]; });
            
            unordered_map<int, int> class_counts;
            for (int i = 0; i < k; ++i) {
                int neighbor_index = indices[i];
                int label = static_cast<int>(train_y[neighbor_index][0]);
                class_counts[label]++;
            }
            int predict_label=-1;
            int max_cnt=-1;
            for (const auto &pair : class_counts) {
                if (pair.second > max_cnt) {
                    max_cnt = pair.second;
                    predict_label = pair.first;
                }
                
            }
            predictions.push_back({static_cast<float>(predict_label)});

        }
        //return vector<vector<float>>();
        return predictions;
    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) {
        // Implement probability estimation for KNN
        vector<unordered_map<int, float>> probabilities;
        
        for (auto &sample : X) {
            vector<float>distances;
            for(int j=0;j<train_X.size();j++){
                float dis=coutDistance(sample,train_X[j]);
                distances.push_back(dis);
                
            }
            vector<int> indices(distances.size());
            iota(indices.begin(), indices.end(), 0);
            partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                [&distances](int i, int j) { return distances[i] < distances[j]; });
            
            unordered_map<int, float> class_probabilities;
            for (int i = 0; i < k; ++i) {
                int neighbor_index = indices[i];
                int label = static_cast<int>(train_y[neighbor_index][0]);
                class_probabilities[label]++;
            }

            for (auto &pair : class_probabilities) {
                pair.second /= k;
            }

            probabilities.push_back(class_probabilities);
        }
        
        return probabilities;
    }

private: 
    // Implement private function or variable if you needed
    int k;
    vector<vector<float>>train_X;
    vector<vector<float>>train_y;
    float coutDistance(const vector<float> &a, const vector<float> &b){
        float distance = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            distance += pow(a[i] - b[i], 2);
        }
        return pow(distance,(float)1/2);
    }

};

class MultilayerPerceptron: public Classifier {
public:
    MultilayerPerceptron(int input_size=77, int hidden_size=77, int output_size=1)
                        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        initializeParameters();
    }
    void fit(vector<vector<float>> &X, vector<vector<float>> &y) override {
        // Implement training logic for MLP including forward and backward propagation
        learning_rate = 0.05;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            cout<<"epoch:"<<epoch<<"/"<<epochs<<endl;
            
            learning_rate=(20*learning_rate)/(20+epoch);
            
            for (size_t i = 0; i < X.size(); ++i) {
                vector<vector<float>> input = {X[i]};
                vector<vector<float>> target = {y[i]};

                _forward_propagation(input);
                _backward_propagation(target);
              
            } 
        }
       
    }
    vector<vector<float>> predict(vector<vector<float>> &X) override {
        // Implement prediction logic for MLP
        vector<vector<float>> predictions;
        for (auto &sample : X) {
            vector<vector<float>> input = {sample};

            _forward_propagation(input); 
            float threshold = 0.5;
            vector<float>predicted_label;
            predicted_label.push_back((outputLayer[0][0]> threshold) ? 1 : 0);
        
            predictions.push_back(predicted_label);
            
        }
        return predictions;
        //return vector<vector<float>>();
    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) {
        // Implement probability estimation for MLP
        vector<unordered_map<int, float>> probabilities;
        for (auto &sample : X) {
            vector<vector<float>> input = {sample};

            // Forward Propagation
            _forward_propagation(input);

            unordered_map<int, float> class_probabilities;
            
            class_probabilities[0] = 1-outputLayer[0][0];
            class_probabilities[1] = outputLayer[0][0];
            

            probabilities.push_back(class_probabilities);
        }
        return probabilities;
        //return vector<unordered_map<int, float>>();
    }
    


private: 
    // Implement private function or variable if you needed
    int input_size;
    int hidden_size;
    int output_size;
    int epochs;
    float learning_rate;

    vector<vector<float>> inputLayer;
    vector<vector<float>> hiddenLayer;
    vector<vector<float>> outputLayer;

    vector<vector<float>> inputWeights;
    vector<vector<float>> hiddenWeights;
    
   

    void initializeParameters() {
        epochs=10;
        learning_rate = 0.05;

        inputWeights = vector<vector<float>>(input_size, vector<float>(hidden_size));
        hiddenWeights = vector<vector<float>>(hidden_size, vector<float>(output_size));
        
        default_random_engine generator;
        normal_distribution<float> distribution(0.0, 1.0);
        
        
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                inputWeights[i][j]= distribution(generator);
            }
        }

        for (int i = 0; i < hidden_size; ++i) {
           
            for (int j = 0; j < output_size; ++j) {
                hiddenWeights[i][j]= distribution(generator);
            }
        }

    }
    
   
   void _forward_propagation(vector<vector<float>> &X) {
        // Implement forward propagation for MLP
        inputLayer = X;
        hiddenLayer = activate(dotProduct(inputLayer, inputWeights));
        outputLayer = activate(dotProduct(hiddenLayer, hiddenWeights));
        
    }
    
    void _backward_propagation(vector<vector<float>> &target) {
        // Implement backwardpropagation for MLP

        vector<vector<float>> out_error;
        vector<float>out_error_temp;
        
        out_error_temp.push_back(outputLayer[0][0]-target[0][0]);        
        out_error.push_back(out_error_temp);
        

        vector<vector<float>> outputGradient = multiply(out_error,(1-outputLayer[0][0])*outputLayer[0][0]);
        hiddenWeights = subtract(hiddenWeights, multiply(dotProduct(transpose(hiddenLayer),outputGradient), learning_rate));

        vector<vector<float>> hiddenGradient = dotProduct(outputGradient, transpose(hiddenWeights));
        for(int i=0;i<hidden_size;i++){
            hiddenGradient[0][i]*=hiddenLayer[0][i]*(1-hiddenLayer[0][i]);
        }
        inputWeights = subtract(inputWeights, multiply(dotProduct(transpose(inputLayer), hiddenGradient), learning_rate*10));

    }
   
   vector<vector<float>> activate(const vector<vector<float>> &matrix) {
        vector<vector<float>> result = matrix;  
        
        for (auto &row : result) {
            for (float &val : row) {
                val = 1.0 / (1.0 + exp(-val));
            }
        }
        
        return result;
    }

    vector<vector<float>> dotProduct(const vector<vector<float>> &A, const vector<vector<float>> &B) {
        size_t m = A.size();
        size_t n = B[0].size();
        size_t p = B.size();

        vector<vector<float>> result(m, vector<float>(n, 0.0));

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < p; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                    
                }
            }
        }

        return result;
    }

    vector<vector<float>> transpose(const vector<vector<float>> &matrix) {
        // Transpose a matrix
        size_t m = matrix.size();
        size_t n = matrix[0].size();

        vector<vector<float>> result(n, vector<float>(m, 0.0));

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    vector<vector<float>> subtract(const vector<vector<float>> &A, const vector<vector<float>> &B) {
        size_t m = A.size();
        size_t n = A[0].size();

        vector<vector<float>> result(m, vector<float>(n, 0.0));

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result[i][j] = A[i][j] - B[i][j];
            }
        }

        return result;
    }

    vector<vector<float>> multiply(const vector<vector<float>> &A, float scalar) {
        size_t m = A.size();
        size_t n = A[0].size();

        vector<vector<float>> result(m, vector<float>(n, 0.0));

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result[i][j] = A[i][j] * scalar;
            }
        }

        return result;
    }
    

};


unordered_map<string, float> evaluate_model(Classifier*, vector<vector<float>>&, vector<vector<float>>&, float);

int main() {
    
    string train_pth = "trainWithLabel.csv";
    string test_pth = "testWithoutLabel.csv";
    //vector<vector<float>> train_df = read_csv_file(train_pth);
    //vector<vector<float>> test_df = read_csv_file(test_pth);
    vector<vector<float>> train_df = read_csv_file(train_pth);
    vector<vector<float>> test_df = read_csv_file(test_pth);
    
    /*create dictionary of models for iterating*/
    unordered_map<float, Classifier*> models;
    models[1.0f] = new NaiveBayesClassifier(); // 1.0 represents Naive Bayes
    models[2.0f] = new KNearestNeighbors(); // 2.0 represents KNN 
    models[3.0f] = new MultilayerPerceptron(); // 3.0 represents MLP 
    

    /*preprocessing*/
    
    //cout<<train_df.size()<<" "<<train_df[0].size()<<endl;
    //cout<<test_df.size()<<" "<<test_df[0].size()<<endl;
    
    IsVal=0;
    Preprocessor test_preprocessor(test_df);
    //train_df = train_preprocessor.preprocess();
    test_df = test_preprocessor.preprocess();
    

    /*split the dataset*/
    vector<vector<float>> X_train = get_X(train_df), y_train = get_y(train_df);
    vector<vector<float>> X_test = get_X_test(test_df);
    
    
    /*k fold cross-validation*/
    int n_splits = 10, random_state = 42;
    vector<PVV> folds = k_fold_split(X_train, y_train, n_splits, random_state);
    vector<vector<unordered_map<string, float>>> cv_result;
    
    for (auto &p: models) {
        Classifier* model = p.second;
        float model_label = p.first;
        vector<unordered_map<string, float>> fold_result;

        for (int fold = 0; fold < folds.size(); fold++) {
            
            cout<<"model: "<<model_label<<"  fold: "<<fold<<endl;
            auto &train_indices = folds[fold].first;
            auto &val_indices = folds[fold].second;
            // get X_fold, y_fold
            vector<vector<float>> X_train_fold, y_train_fold, X_val_fold, y_val_fold;

            for (auto &idx: train_indices) {
                X_train_fold.push_back(X_train[idx]);
                y_train_fold.push_back(y_train[idx]);
            }
            IsVal=0;
            Preprocessor train_preprocessor(X_train_fold);
            X_train_fold = train_preprocessor.preprocess();

            for (auto &idx: val_indices) {
                X_val_fold.push_back(X_train[idx]);
                y_val_fold.push_back(y_train[idx]);
            }
            IsVal=1;
            Preprocessor val_preprocessor(X_val_fold);
            X_val_fold = val_preprocessor.preprocess();


            model->fit(X_train_fold, y_train_fold);
            unordered_map<string, float> res = evaluate_model(model, X_val_fold, y_val_fold, model_label);
            fold_result.push_back(res);
        }
        cv_result.push_back(fold_result);
    }
    write_result_to_csv(cv_result);

    unordered_map<float, vector<vector<float>>> predictions;

    for (auto &p: models) {
        Classifier* model = p.second;
        predictions[p.first] = model->predict(X_test);
    }
    write_predictions_to_csv(predictions);
    cout << "Model predictions saved to test_results.csv\n";

    
    return 0;
}

unordered_map<string, float> evaluate_model(Classifier* model, vector<vector<float>>& X_test, vector<vector<float>>& y_test, float model_label) {
    vector<vector<float>> predictions = model->predict(X_test);
    vector<unordered_map<int, float>> proba_array = model->predict_proba(X_test);

    float accuracy = accuracy_score(y_test, predictions);
    float precision = precision_score(y_test, predictions);  // positive label = 1
    float recall = recall_score(y_test, predictions);
    float f1 = f1_score(recall, precision);
    float mcc = matthews_corrcoef(y_test, predictions);
    double auc = roc_auc_score(y_test, proba_array);

    return {{"model", model_label}, {"accuracy", accuracy}, {"f1", f1}, {"precision", precision},
            {"recall", recall}, {"mcc", mcc}, {"auc", auc}};
}
