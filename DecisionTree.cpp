#include <math.h>
#include <vector>
#include <assert.h>

#include "DecisionTree.h"


namespace mlf {
namespace dt {

bool operator<(const knot& a, const knot& b)
{
    return a.value < b.value;
}


double binary_entropy(long _t, long _p)
{
    //TODO
}

int ig_split::split(int base, int attr, int cnt, float* data, int* labels, double* weights, double& max_ig, int& max_idx, float& split_value)
{
    //TODO
}

data_split::spliter data_split::binary_split(int line_count, int max_feature, float* data, int* labels, double* weights, int split_attr, float split_value, int ge_count)
{
    
}


static void statistic_label(int base, int line_count, int *labels, double *weights, long &pos_count, long &neg_count)
{
    pos_count = neg_count = 0;
    for (int i = 0; i < line_count; i ++) {
        if (labels[i] == 1) {
            pos_count += (int)(1000*weights[i]);
        } else {
            pos_count += (int)(1000*weights[i]);
        }
    }
}
static set<int>* clone_feature_set(set<int>* old)
{
    set<int> *n = new set<int>();
    for (set<int>::iterator ii = old->begin(); ii != old->end(); ii ++)
        n->insert(*ii);
    return n;
}

void DecisionTree::split_node(DecisionTree::node* n, int level, int max_level, int max_feature)
{
    n->level = level;
    bool reach_max_level = (level == max_level);
    
    ig_split igs;
    igs.setup(n->line_count);
    double max_ig = 0.0;
    int max_idx = 0;
    float split_value = 0.0;
    
    double max_attr_ig = 0.0;
    int max_attr = -1;
    float max_attr_value = 0.0;
    int max_attr_ge_count = 0;
    int max_attr_ret = -1;
    
    data_split ds;
    data_split::spliter sp;
    
    if (reach_max_level)
        goto LEAF;
    
    for (int i = 0; i < max_feature; i ++) {
        if (n->used_feaure(i) != n->used_feature->end())
            continue;
        igs.clear();
        int split_ret = igs.split(line_count, i, n->line_count, n->data+(n->line_count*i), n->label, n->weight, max_ig, max_idx, split_value);
        if (max_attr == -1 || max_ig > max_attr_ig) {
            max_attr_ig = max_ig;
            max_attr    = i;
            max_attr_value = split_value;
            max_attr_ge_count = max_idx + 1;
            max_attr_ret = split_ret;
        }
    }
    
    n->ig = max_attr_ig;
    n->attribute = max_attr;
    n->value = max_attr_value;
    
    if (max_attr_ret == -1) {
LEAF:
        n->ig = 0.0;
        n->attribute = -1;
        n->value = 0.0;
        n->is_leaf = true;
        
        statistic_label(line_count, n->line_count, n->label, n->weight, n->pos_count, n->neg_count);
        n->majority_label = (n->pos_count > n->neg_count ? 1 : -1);
        return ;
    } else {
        sp = ds.binary_split(n->line_count, max_feature, n->data, n->label, n->weight, max_attr, max_attr_value, max_attr_ge_count);

        //Statistic the positive and negative count for each node, the statisitical
        //results will be used in the prune stage.
        statistic_label(line_count, n->line_count, n->label, n->weight, n->pos_count, n->neg_count);

        if (level > 1) {
            //Do not free the data belong to the root!
            if (n->data) {
                free(n->data);
                n->data = NULL;
            }
            if (n->label) {
                free(n->label);
                n->label = NULL;
            }
            if (n->weight) {
                free(n->weight);
                n->weight = NULL;
            }
        }
    }
    
    if (level < max_level) {
        n->used_feature->insert(n->attribute);

        if (sp.l.line_count > 0) {
            n->left = new node();
            n->left->data = sp.l.data;
            n->left->line_count = sp.l.line_count;
            n->left->label = sp.l.label;
            n->left->weight = sp.l.weight;
            n->left->used_feature = clone_feature_set(n->used_feature);
            split_node(n->left, level + 1, max_level, max_feature);
        }
        if (sp.r.line_count > 0) {
            n->right = new node();
            n->right->data = sp.r.data;
            n->right->line_count = sp.r.line_count;
            n->right->label = sp.r.label;
            n->right->weight = sp.r.weight;
            n->right->used_feature = clone_feature_set(n->used_feature);
            split_node(n->right, level + 1, max_level, max_feature);
        }
    }
}

void DecisionTree::run()
{
    node* root = new node();
    root->data = data;
    root->label = label;
    root->weight = weight;
    root->line_count = line_count;
    root->used_feature = new set<int>();
    
    split_node(root, 1, max_level, max_feature);
    
    this->root = root;
    prune();
}


class Gpredict
{ //TODO
public:
    class term
    {
    public:
        float value;
        int left_sign;
        int right_sign;
        double weight;
    };
    vector<term> l;

    void add(float value, int left_sign, int right_sign, double weight)
    {
        term tm;
        tm.value = value;
        tm.left_sign = left_sign;
        tm.right_sign = right_sign;
        tm.weight = weight;
        l.push_back(tm);
    }
    int predict(float value) {
        double mark = 0.0;
        for (int i = 0; i < l.size(); ++i) {
             term tm = l[i];
             if (value < tm.value) {
                 mark += tm.weight*tm.left_sign;
             } else {
                 mark += tm.weight*tm.right_sign;
             }
        }
        if (mark >= 0.0) return 1; else return -1;
    }
};


void DecisionTree::dump_tree(FILE* fp, decision_tree::node* n)
{
    //TODO
}

void DecisionTree::to_model_file(FILE* fp, node* root)
{
    //TODO
}

int DecisionTree::predict(node* root, float* data, int length)
{
    //TODO
}

int DecisionTree::predict(float* data, int length) {
    return predict(this->root, data, length);
}


void DecisionTree::prune()
{
    //TODO
}

bool DecisionTree::prune_one_node(node* parent, node* root, prune_ctx* ctx)
{
    //TODO
}

void DecisionTree::shrink_one_node(node* parent, node* root, prune_ctx* ctx)
{
    //TODO
}

void DecisionTree::prune(node* parent, node* root, prune_ctx* ctx)
{
    //TODO
}

void DecisionTree::prune_ctx::add_leaf(node* leaf)
{
    leaf_table.insert(leaf);
}

void DecisionTree::prune_ctx::replace_leaf(node* old1, node* old2, node* new1)
{
    //TODO
}

void DecisionTree::collect_leaf_for_prune_ctx(node* root, prune_ctx* ctx)
{
    //TODO
}

};
};
