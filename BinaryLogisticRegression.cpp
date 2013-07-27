#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <float.h>
#include <map>
#include <fstream>
#include <stdio.h>
using namespace std;

const bool debug = true;
const int MAX_FEATURE_NUM = 100;
const int MAX_TRAIN_NUM   = 50000;
namespace utils 
{
    bool split(string str, const char delim, vector<string> &result)
    {
        result.clear();
        while (true)
        {
            int pos = str.find(delim);
            if (pos == 0)
            {
                str = str.substr(1);
                continue;
            }
            if (pos < 0)
            {
                result.push_back(str);
                break;
            }
            result.push_back(str.substr(0, pos));
            str = str.substr(pos+1);
        }
        return true;
    }
};



namespace feature_scale
{
    struct feature_stat_s
    {
        double min, max, total, avg;
        int count;
        feature_stat_s()
        {
            min = DBL_MAX;
            max = DBL_MIN;
            total = avg = 0.0f;
            count = 0;
        }
    };
    static feature_stat_s feature_stats[MAX_FEATURE_NUM];
    static int feature_num;

    /**
     * data[m][n]
     * m : train set number
     * n : feature number
     */
    void init_feature_stat(double data[][MAX_FEATURE_NUM], int m, int n)
    {
        feature_num = n;
        for (int i = 0; i < n; i ++)
        {
            for (int j = 0; j < m; j ++)
            {
                feature_stats[i].count ++;
                feature_stats[i].total += data[j][i];
                if (feature_stats[i].max < data[j][i]) feature_stats[i].max = data[j][i];
                if (feature_stats[i].min > data[j][i]) feature_stats[i].min = data[j][i];
            }
            feature_stats[i].avg = feature_stats[i].total / feature_stats[i].count;
        }
    }

    /**
     * data[m][n]
     * m : train set number
     * n : feature number
     */
    void scaling(double data[][MAX_FEATURE_NUM], int m, int n)
    {
        for (int i = 0; i < n; i ++)
        {
            double range = feature_stats[i].max - feature_stats[i].min;
            double ratio = 0.0f;
            bool   deleted = false;
            
            for (int j = 0; j < m; j ++)
            {
                if (range != 0.0f)
                {
                    data[j][i] = (data[j][i] - feature_stats[i].avg) / range;
                }
                else
                {
                    data[j][i] = 0.0f;
                    deleted = true;
                }
            }
        }
    }
    /**
     * data[n]
     * n : feature number
     */
    void scaling(double data[MAX_FEATURE_NUM], int n)
    {
        for (int i = 0; i < n; i ++)
        {
            double range = feature_stats[i].max - feature_stats[i].min;
            double ratio = 0.0f;
            double allRowsVar = 0.0f;
            bool   deleted = false;
            
                if (range != 0.0f)
                {
                    data[i] = (data[i] - feature_stats[i].avg) / range;
                }
                else
                {
                    data[i] = 0.0f;
                    deleted = true;
                }
        }
    }
};

namespace logistic_regression
{
    const  double default_learning_rate = 10;
    const  int    default_learning_loop = 50;
    static double trainX[MAX_TRAIN_NUM][MAX_FEATURE_NUM];
    static double trainY[MAX_TRAIN_NUM];
    static int    train_number;
    static int    feature_number;

    double theta[MAX_FEATURE_NUM];
    double hmatrix[MAX_TRAIN_NUM];

    namespace sigmoid_func
    {
        void sigmoid(double *z[], int m, int n)
        {
            for (int i = 0; i < m; i ++)
                for (int j = 0; j < n; j ++)
                    z[i][j] = 1.0f / (1.0f + exp(-z[i][j]));
        }
        void sigmoid(double z[], int n)
        {
            sigmoid(&z, 1, n);
        }
        double sigmoid(double z)
        {
            return 1.0f / (1.0f + exp(-z));
        }
    };

    namespace hypothesis_func
    {
        /**
         * theta[n] 
         * data[m][n]
         * result[m]
         * 
         * m train number
         * n feature number
         * result[i] = sigmoid(SUM(theta[j]*data[i][j]))
         */
        void hypothesis(double theta[], double data[][MAX_FEATURE_NUM], double result[], int m, int n)
        {
            for (int i = 0; i < m; i ++)
            {
                result[i] = 0.0f;
                for (int j = 0; j < n; j ++)
                {
                    result[i] += theta[j] * data[i][j];
                }
                result[i] = sigmoid_func::sigmoid(result[i]);
            }
        }
        /**
         * theta[n] 
         * data[n]
         * result
         * 
         * n feature number
         * result = sigmoid(SUM(theta[j]*data[i][j]))
         */
        double hypothesis(double theta[], double data[], int n)
        {
            double score = 0.0f;
            for (int i = 0; i < n; i ++)
                score += theta[i] * data[i];
            return sigmoid_func::sigmoid(score);
        }
    };

    namespace cost_func
    {
        /**
         * h[m] y[m] theta[m]
         */
        double cost_func(double h[], double y[], double theta[], int m, int n)
        {
            double cost = 0.0f;
            for (int i = 0; i < n; i ++)
            {
                cost -= (y[i]*log(h[i]) + (1-y[i])*log(1-h[i]));
            }
            cost /= n;
            /**
            if (lamda != 0)
            {
                double sum = 0.0f;
                for (int i = 0; i < m; i ++)
                {
                    sum += theta[i] * theta[i];
                }
                cost += (sum / (2*m))*lamda;
            }*/
            return cost;
        }
    }

    namespace train_func
    {
        void train(double alpha, int max_loop)
        {
            for (int i = 0; i < feature_number; i ++) theta[i] = 0.0f;
            for (int loop = 0; loop < max_loop; loop ++)
            {
                hypothesis_func::hypothesis(theta, trainX, hmatrix, train_number, feature_number);
                double cost = cost_func::cost_func(hmatrix, trainY, theta, train_number, feature_number);
                for (int j = 0; j < feature_number; j ++)
                {
                    double sum = 0;
                    for (int i = 0; i < train_number; i ++)
                    {
                        sum += (hmatrix[i] - trainY[i]) * trainX[i][j];
                    }
                    theta[j] = theta[j] - sum * alpha / train_number;
                }
                hypothesis_func::hypothesis(theta, trainX, hmatrix, train_number, feature_number);
                double newcost = cost_func::cost_func(hmatrix, trainY, theta, train_number, feature_number);
                if (newcost > cost) alpha *= 0.75f;
            }
        }
    }

    namespace predict
    {
        double predict(double data[], int n)
        {
            double score = hypothesis_func::hypothesis(theta, data, n);
            return score;
        }
    };
};

int main(int argc, char *argv[])
{
    double learning_rate = logistic_regression::default_learning_rate;
    int    learning_loop = logistic_regression::default_learning_loop;
    if (argc == 3)
    {
        sscanf(argv[1], "%lf", &learning_rate);
        sscanf(argv[2], "%d",  &learning_loop);
    }
    vector<string> fields;
    string line;
    int y, fid;
    double val;
    
    scanf("%d %d\n", &logistic_regression::train_number, 
        &logistic_regression::feature_number);
    logistic_regression::feature_number ++;
    for (int n = 0; n < logistic_regression::train_number; n ++)
    {
        getline(cin, line);
        utils::split(line, ' ', fields);
        
        logistic_regression::trainY[n] = 0;
        if (fields[1] == "+1") logistic_regression::trainY[n] = 1;
        
        logistic_regression::trainX[n][0] = 1.0f;
        for (int i = 2; i < (int)fields.size(); i ++)
        {
            sscanf(fields[i].c_str(), "%d:%lf", &fid, &val);
            logistic_regression::trainX[n][fid] = val;
        }
    }

    feature_scale::init_feature_stat(logistic_regression::trainX, 
        logistic_regression::train_number, logistic_regression::feature_number);
    feature_scale::scaling(logistic_regression::trainX, 
        logistic_regression::train_number, logistic_regression::feature_number);
    logistic_regression::train_func::train(learning_rate, learning_loop);
    
    int Q;
    scanf("%d\n", &Q);
    double feature[MAX_FEATURE_NUM];
    while (Q --)
    {
        getline(cin, line);
        utils::split(line, ' ', fields);
        feature[0] = 1.0f;
        for (int i = 1; i < (int)fields.size(); i ++)
        {
            sscanf(fields[i].c_str(), "%d:%lf", &fid, &val);
            feature[fid] = val;
        }
        feature_scale::scaling(feature, logistic_regression::feature_number);
        double score = logistic_regression::predict::predict(feature, logistic_regression::feature_number);
        if (score >= 0.5f) fprintf(stdout, "%s +1\n", fields[0].c_str());
        else fprintf(stdout, "%s -1\n", fields[0].c_str());
    }

    return 0;
}
