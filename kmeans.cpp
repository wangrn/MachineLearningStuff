#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>

using namespace std;

const int CLUSTER_NUM = 3;      //< 聚类的数量
const int MAX_ITER_NUM = 30;    //< 最大迭代次数

typdef vector<double> Tuple;    //< 存储每个样例的属性信息

int SampleNumber;       //< 数据集中的样例数目
int DimensionsNumber;   //< 每个样例的维数
vector<Tuple> tuples;   //< 数据集


//计算两个元祖的欧几里距离
double getDistXY(const Tuple &x, const Tuple &y)
{
    double sum = 0;
    for (int i = 0; i < DimensionsNumber; i ++)
        sum += (x[i]-y[i])*(x[i]-y[i]);
    return sqrt(sum);
}

// 
int clusterOfTuple(Tuple means[], const Tuple &tuple)
{
    double dist = getDistXY(means[0], tuple);
    int label = 0;
    for (int i = 1; i < CLUSTER_NUM; i ++)
    {
        double tmp = getDistXY(means[i], tuple);
        if (tmp < dist)
        {
            dist = tmp;
            label = i;
        }
    }
    return label;
}


int main()
{
    printf("Hello world!\n");
    return 0;
}
