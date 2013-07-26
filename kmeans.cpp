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

int SampleNumber;   //< 数据集中的样例数目
int DimensionsNumber;      //< 每个样例的维数

int main()
{
    printf("Hello world!\n");
    return 0;
}
