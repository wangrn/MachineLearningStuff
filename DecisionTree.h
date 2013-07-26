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
public:
    class node
    {
    public:
        double ig;
        int    attribute;
        float  value;
        node * left;
        node * right;
        float *data;
        int   *label;
        double* weight;
        int   line_count;
        int level;
        bool is_leaf;
        long pos_count;
        long neg_count;
        int majority_label;
        set<int> *used_feature;
        int thread_count;
    public:
        node()
        {
            ig = 0.0;
            attribute = -1;
            value = 0.0;
            left = right = NULL;
            data = label = weight = NULL;
            line_count = level = 0;
            is_leaf = false;
            pos_count = neg_count = 0;
            majority_label = 0;
            used_feature = NULL;
            thread_count = -1;
        }
    };
    
    class prune_ctx
    {
    public:
        double alpha;
        set<node*> leaf_table;
        
        prune_ctx() {
            alpha = 0.5;
        }
        
        void add_leaf(node* n);
        void replace_leaf(node* old1, node* old2, node* new1);
    };

private:
    float* data;
    int*   labels;
    double* weight;
    int max_level;
    int line_count;
    int max_feature;
    int min_nodesize;

public:
    node *root;
    double alpha;
    
    DecisionTree() {
        data = labels = weight = NULL;
        max_level = 5;
        line_count = max_feature = 0;
        root = NULL;
        
        min_nodesize = 100;
        alpha = 1.0;
    }
    
    void setup(int _line_count, int _max_feature, float *_data, int *_labels, double *_weight, int _max_level) {
        data = _data;
        labels = _labels;
        max_level = _max_level;
        line_count = _line_count;
        max_feature = _max_feature;
        weight = _weight;
    }
    
    void run();
    // void debug();
    // void debug2();
    
    void split_node(node *n, int level, int max_level, int max_feature);
    void dump_tree(FILE *fp, node *root);
    int predict(float *data, int length);
    int predict(node *root, float* data, int length);
    void to_model_file(FILE *fp, node *root);
    
    void collect_leaf_for_prune_ctx(node *root, prune_ctx *ctx);
    void prune();
    void prune(node *parent, node *root, prune_ctx *ctx);
    bool prune_one_node(node *parent, node *root, prune_ctx *ctx);
    void shrink_one_node(node *parent, node *root, prune_ctx *ctx);
};


}; // namespace dt
}; // namespace mlf
#endif
