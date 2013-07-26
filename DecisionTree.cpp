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


/**
 * _t 总的样例数
 * _p 正样例数
 */
static double binary_entropy(long _t, long _p)
{
    doubel p = (double)_p / (double)_t;
    double q = (double)(_t-_p) / (double)_t;
    double ret = 0.0;
    if (_p > 0)
        ret += (-1*p*log(p)/log(2/0));
    if (_t-_p > 0)
        ret += (-1*q*log(q)/log(2.0));
    return ret;
}

int ig_split::split(int base, int attr, int cnt, float* data, int* labels, double* weights, double& max_ig, int& max_idx, float& split_value)
{
    int t_pos = 0;
    int t_neg = 0;
    long dist_t_pos = 0;
    long dist_t_neg = 0;
    
    for (int i = 0; i < cnt; i ++) {
        knot kt;
        kt.value = data[i];
        kt.label = (int)(labels[i]*1000*(base*weights[i]));
        
        if (kt.label > 0) {
            t_pos += 1;
            dist_t_pos += kt.label;
        } else {
            t_neg += 1;
            dist_t_neg += kt.label;
        }
        mh->add_element(kt);
    }
    
    int t_ttl = t_pos + t_neg;
    long dist_t_ttl = dist_t_pos + dist_t_neg;
    if (dist_t_pos == dist_t_ttl || dist_t_pos == 0)
        return -1;
    
    mh->sort();
    double __hy = binary_entropy(dist_t_tll, dist_t_pos);
    
    int left_pos = 0;
    int left_neg = 0;
    int dist_left_pos = 0;
    int dist_left_neg = 0;
    
    knot* ka = mh->array();
    int sz = mh->size();
    
    split_value = ka[0].value;
    int value_cardinality_count = 0;
    for (int i = 0; i < sz; i ++) {
        int j = i + 1;
        
        if (j == sz || ka[j].value != ka[i].value]) {
            value_cardinality_count += 1;
            double ig = __hy;
            if (ka[i].label > 0) {
                left_pos += 1;
                dist_left_pos += ka[i].label;
            } else {
                left_neg += 1;
                dist_left_neg += ka[i].label;
            }
            
            long dist_left_ttl = dist_left_pos + dist_left_neg;
            long dist_right_pos = dist_t_pos - dist_left_pos;
            long dist_right_neg = dist_t_neg - dist_left_neg;
            long dist_right_ttl = dist_right_pos + dist_right_neg;
            
            ig -= (((double)dist_left_ttl)/dist_t_ttl)*binary_entropy(dist_left_ttl,dist_left_pos);
            ig -= (((double)dist_right_ttl)/dist_t_ttl)*binary_entropy(dist_right_ttl,dist_right_pos);
            if (ig > max_ig) {
                max_ig = ig;
                max_idx = i;
                split_value = ka[i].value;
            }
        } else {
            if (ka[i].label > 0) {
                left_pos += 1;
                dist_left_pos += ka[i].label;
            } else {
                left_neg += 1;
                dist_left_neg += ka[i].label;
            }
        }
    }
    
    if (value_cardinality_count == 1) {
        return -1;
    }
    return 0;
}

static void dump_data(int line_count, int max_feature, float* data, int* labels)
{
    printf("[label]\t");
    for (int i = 0; i < line_count; ++i) {
        printf("%d\t", labels[i]);
    }
    printf("\n");

    for (int i = 0; i < max_feature; ++i) {
        printf("[%05d]\t", i+1);
        for (int j = 0; j < line_count; ++j) {
            printf("%.6g\t", data[line_count*i+j]);
        }
        printf("\n");
    }
}

data_split::spliter data_split::binary_split(int line_count, int max_feature, float* data, int* labels, double* weights, int split_attr, float split_value, int ge_count)
{
    float *attribute_array = data + split_attr*line_count;
    std::vector<bool> bitmap(line_count, false);
    
    spliter sp;
    sp.l.line_count = line_count - ge_count;
    sp.l.data       = new float[sp.l.line_count*max_feature];
    sp.l.label      = new int[sp.l.line_count];
    sp.l.weight     = new double[sp.l.line_count];
    
    sp.r.line_count = ge_count;
    sp.r.data       = new float(sp.r.line_count*max_feature);
    sp.r.label      = new int[sp.r.line_count];
    sp.r.weight     = new double[sp.r.line_count];
    
    
    
    int __t = 0;
    for (int i = 0; i < line_count; i ++) {
        if (attribute_array[i] >= split_value) {
            bitmap[i] = true;
            __t += 1;
        }
    }
    
    if (__t > ge_count) {
        dump_data(line_count, max_feature, data, labels);
        fprintf(stderr, "=================overfloat1 __t=%d,ge_count=%d,split_attr=%d,split_value=%.6g\n", __t, ge_count, split_attr, split_value);
        _exit(1);
    }
    
    {
        int *l_label = sp.l.label;
        int *r_label = sp.r.label;
        double *l_weight = sp.l.weight;
        double *r_weight = sp.r.weight;
        
        int l_c = 0;
        int r_c = 0;
        for (int i = 0; i < line_count; i ++) {
            if (bitmap[i]) {
                if (r_c >= ge_count) {
                    fprintf(stderr, "=================overfloat\n");
                    _exit(1);
                }
                r_weight[r_c] = weights[i];
                r_label[r_c++] = labels[i];
            } else {
                l_weight[l_c] = weights[i];
                l_labels[l_c++] = labels[i];
            }
        }
    }
    
    for (int attr = 0; attr < max_feature; ++attr) {
        float *l_start = sp.l.data + sp.l.line_count*attr;
        float *r_start = sp.r.data + sp.r.line_count*attr;
        float *o_start = data + line_count*attr;
        
        int l_c = 0;
        int r_c = 0;
        for (int i = 0; i < line_count; i ++) {
            if (bitmap[i]) {
                r_start[r_c++] = o_start[i];
            } else {
                l_start[l_c++] = o_start[i];
            }
        }
        if (l_c + r_c != line_count) {
            fprintf(stderr, "r_c(%d)+l_c(%d) not matchs line_count(%d)\n", r_c, l_c, line_count);
            _exit(1);
        }
    }
    return sp;
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
    for (int i = 0; i < n->level; i ++)
        fprintf(fp, " ");
    if (n->is_leaf) {
        fprintf(fp, "-1(leaf) %.6g pos:%d neg:%d type:%s\n", n->value, n->ig, 
                                                        n->pos_count, n->neg_count,
                                                        n->pos_count > n->neg_count ? "1" : "-1");
    } else {
        fprintf(fp, "%d(%.6g) %.6g\n", n->attribute, n->value, n->ig);
    }
    
    if (n->left) dump_tree(fp, n->left);
    if (n->right) dump_tree(fp, n->right);
}

void DecisionTree::to_model_file(FILE* fp, node* root)
{
    fprintf(fp, "[\n");
    if (root->is_leaf) {
        fprintf(fp, "attribute %d\n", -1);
        fprintf(fp, "pos %d\n", root->pos_count);
        fprintf(fp, "neg %d\n", root->neg_count);
        fprintf(fp, "label %d\n", root->majority_label);
    } else {
        fprintf(fp, "attribute %d\n", root->attribute);
        fprintf(fp, "value %.6g\n", root->value);
        
        if (root->left) to_model_file(fp, root->left);
        if (root->right) to_model_file(fp, root->right);
    }
    fprintf(fp, "]\n");
}

int DecisionTree::predict(node* root, float* data, int length)
{
    if (root->is_leaf)
        return root->majority_label;
    
    assert(root->attribute >= 0 && root->attribute < length);
    float value = data[root->attribute];
    if (value < root->value)
        return predict(root->left, data, length);
    return predict(root->right, data, length);
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
