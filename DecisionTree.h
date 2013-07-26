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
}

};
};
#endif
