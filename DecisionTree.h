#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <set>
using namespace std;

#include "MinHeap.h"


namespace mlf {
namespace dt {

class knot
{
public:
    float value;
    int   label;
}

class ig_split
{
public:
    utils::min_heap<knot>* mh;

public:
    ig_split() {
        mh = new utils::min_heap<knot>();
    }
    void setup(int max_feature) {
        mh->init(max_feature);
    }
    void clear() {
        mh->clear();
    }
    
    int split(int base, int attr, int cnt, float* data, int* labels, double* weights, double& max_ig, int& max_idx, float& split_value);
};

class data_split
{
public:
    class block
    {
    public:
        float *data;
        int   *label;
        double *weight;
        int   line_count;
        
        block() {
            data = NULL;
            lebel = NULL;
            line_count = 0;
        }
        void clone(int lc, int max_feature, float *_data, int* _label, double *_weight) {
            line_count = lc;
            data = new float[line_count*max_feature];
            memcpy(data, _data, line_count*max_feature*sizeof(float));
            label = new int[line_count];
            memcpy(label, _label, line_count*sizeof(int));
            weight = new double[line_count];
            memcpy(weight, _weight, line_count*sizeof(double));
        }
    };
    
    class spliter
    {
    public:
        block l;
        block r;
    };
    
    spliter binary_split(int line_count, int max_feature, float *data, int *labels, double *weights, int split_attr, float split_value, int ge_count);
};

class DecisionTree
{
    
};


}; // namespace dt
}; // namespace mlf
#endif
