#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>

using namespace std;

const int CLUSTER_NUM = 3;      //< 聚类的数量
const int MAX_ITER_NUM = 30;    //< 最大迭代次数

typedef vector<double> Tuple;    //< 存储每个样例的属性信息

int SampleNumber;       //< 数据集中的样例数目
int DimensionsNumber;   //< 每个样例的维数
vector<Tuple> tuples;   //< 数据集


//计算两个元祖的欧几里距离
double getDistXY(const Tuple &x, const Tuple &y)
{
    double sum = 0;
    for (int i = 1; i <= DimensionsNumber; i ++)
        sum += (x[i]-y[i])*(x[i]-y[i]);
    return sqrt(sum);
}

// 根据质心, 决定当前tuple属于哪个cluster
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

// 获得给定cluster的平方误差
double getVar(vector<Tuple> clusters[], Tuple means[]) 
{
    double var = 0;
    for (int i = 0; i < CLUSTER_NUM; i ++)
    {
        vector<Tuple> t = clusters[i];
        for (size_t j = 0; j < t.size(); j ++)
        {
            var += getDistXY(t[j], means[i]);
        }
    }
    return var;
}


//获得当前簇的均值（质心）
Tuple getMeans(const vector<Tuple>& cluster)
{
    int num = cluster.size();
    Tuple t(DimensionsNumber+1, 0);
    for (int i = 0; i < num; i++)
    {
        for(int j = 1; j <= DimensionsNumber; ++j)
        {
            t[j] += cluster[i][j];
        }
    }
    for(int j = 0; j <= DimensionsNumber; ++j)
        t[j] /= num;
    return t;
}

void print(const vector<Tuple> clusters[])
{
    for(int lable = 0; lable < CLUSTER_NUM; lable ++)
    {
        cout << "Cluster#" << lable << endl;
        vector<Tuple> t = clusters[lable];
        for(size_t i = 0; i < t.size(); i ++)
        {
            cout<<i<<".(";
            for(int j = 0; j <= DimensionsNumber; ++j)
            {
                cout<<t[i][j]<<", ";
            }
            cout<<")\n";
        }
    }
}


void KMeans(vector<Tuple>& tuples)
{
    vector<Tuple> clusters[CLUSTER_NUM];
    Tuple means[CLUSTER_NUM];//中心点
    
    int i = 0;
    //一开始随机选取k条记录的值作为k个簇的质心（均值）
    srand((unsigned int)time(NULL));
    for(i = 0; i < CLUSTER_NUM;)
    {
        int iToSelect = rand()%tuples.size();
        if(means[iToSelect].size() == 0)
        {
            for(int j = 0; j <= DimensionsNumber; ++ j)
            {
                means[i].push_back(tuples[iToSelect][j]);
            }
            ++i;
        }
    }
    
    int lable = 0;
    //根据默认的质心给簇赋值
    for(i = 0; i != tuples.size(); ++i)
    {
        lable = clusterOfTuple(means, tuples[i]);
        clusters[lable].push_back(tuples[i]);
    }
    
    double oldVar = -1;
    double newVar = getVar(clusters, means);
    cout << "初始的的整体误差平方和为：" << newVar << endl;
    int t = 0;
    double dif = newVar - oldVar;
    while((dif >= 1.0 || dif <= -1) && t<MAX_ITER_NUM) //当新旧函数值相差不到1即准则函数值不发生明显变化时，算法终止
    {
        cout<<"第 "<<++t<<" 次迭代开始："<<endl;
        for (i = 0; i < CLUSTER_NUM; i++) //更新每个簇的中心点
        {
            means[i] = getMeans(clusters[i]);
        }
        oldVar = newVar;
        newVar = getVar(clusters,means);
        dif = newVar - oldVar;
        
        for (i = 0; i < CLUSTER_NUM; i++)
        {
            clusters[i].clear();
        }
        //根据新的质心获得新的簇
        for(i = 0; i != tuples.size(); ++ i)
        {
            lable = clusterOfTuple(means, tuples[i]);
            clusters[lable].push_back(tuples[i]);
        }
        cout<<"此次迭代之后的整体误差平方和为："<<newVar<<endl;
    }
    
    cout << "The result is:\n";
    print(clusters);
}
int main()
{
    char fname[256];
    cout << "请输入存放数据的文件名： ";
    cin >> fname;
    cout << endl <<" 请依次输入: 维数 样本数目" << endl;
    cout << endl <<" 维数dimNum: ";
    cin >> DimensionsNumber;
    cout << endl <<" 样本数目SampleNumber: ";
    cin >> SampleNumber;
    
    ifstream infile(fname);
    if(!infile)
    {
        cout<<"不能打开输入的文件"<<fname<<endl;
        return 0;
    }
    
    tuples.clear();
    //从文件流中读入数据
    for(int i = 0; i < SampleNumber && !infile.eof(); ++i)
    {
        string str;
        getline(infile, str);
        istringstream istr(str);
        Tuple tuple(DimensionsNumber+1, 0);//第一个位置存放记录编号，第2到dimNum+1个位置存放实际元素
        tuple[0] = i+1;
        for(int j = 1; j <= DimensionsNumber; ++j)
        {
            istr >> tuple[j];
        }
        tuples.push_back(tuple);
    }
    
    cout<<endl<<"开始聚类"<<endl;
    KMeans(tuples);
    
    return 0;
}
