/**
 *
 * @author : wangrn
 * 
 * 改进的朴素贝叶斯算法, 结合词在类别中的权重, 进行适当的调权.
 * 
 * 
 * Pr(c|d) 是文档d属于类c的概率. 假定文档内词是独立的, 则Pr(c|d) = Σ Pr(c|w)*Pr(w|d) .
 *      Pr(w|d)是词w在文档d中的词频, 可以是各种归一化或平滑公式.
 * 
 * Pr(c|w) = Pr(w|c) * Pr(c)
 *      Pr(c) 先验概率, 类c的样本数, 可以从训练集中统计出. 
 * 
 * Pr(w|c) = Pow( N(w|c)/N(w), 1.5) * factor
 *      N(w|c)是词w在类c中的词频, N(w)是词w在所有训练集中的词频.
 *      factor为调权因子, 根据类c下的词分布, 进行调权, 有两种情况:
 *              1. N(w|c) >= AVG(c) , factor = 1.0
 *              2. else, factor = Pow( N(w|c)/AVG(c) , alpha)
 *              说明:
 *                  a) AVG(c)是类c下所有词的平均词频
 *                  b) alpha调权因子, 默认为0.2
 * 
 * 其他注意:
 *      1. 计算AVG(c)时, 丢弃掉N(w|c)特别小的
 *      2. 丢弃 Pr(w|c)特别小的, 默认阈值0.1
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <map>

using namespace std;


//改进的朴素贝叶斯, 模型公式参见文件开头注释.
class NaiveBayesianClassifier
{
public:
    NaiveBayesianClassifier()
    {
        alpha = 0.2;
        min_count_in_class = 50.0;
        threshold = 0.1;
        min_count = 3;
    }
    
    void train(FILE *input, FILE *output)
    {
        char line[4096];
        char *label, *idx, *val, *endptr;
        int total = 0;
        double value = 0;
        
        while (fgets(line, sizeof(line), input))
        {
            int i = 0;
            label = strtok(line, "\t");
            if (label == NULL) continue;
            
            Cj[label] ++;
            while (true)
            {
                idx = strtok(NULL, ":");
                if (idx == NULL) break;
                val = strtok(NULL, "\t");
                if (val == NULL) break;
                
                errno = 0;
                value = strtod(val, &endptr);
                if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                    continue;
                
                DFi[idx] ++;
                DFij[idx][label] ++;
                mClassWordDF[label][idx] ++;
                
                i++;
            }
            total ++;
        }
         
        for (map<string, map<string, uint32_t> >::iterator it = mClassWordDF.begin();
                          it != mClassWordDF.end(); it ++)
        {
            float pr = 0.0;
            int k = 0;
            for (map<string, uint32_t>::iterator it1 = it->second.begin();
                              it1 != it->second.end(); it1 ++)
            {
                if (it1->second > min_count_in_class)
                {
                    k ++;
                    pr += it1->second;
                }
            }
            if (k > 0) mClassAvgDF[it->first] = pr / k;
        }
        
        float weight, factor;
        for (map<string,uint32_t>::iterator it1=DFi.begin();it1!=DFi.end();it1++)
        {
            if (it1->second > min_count)
            {
                map<string, uint32_t> iDFj = DFij[it1->first];
                for(map<string,uint32_t>::iterator it2=iDFj.begin();it2!=iDFj.end();it2++)
                {
                    factor = 1.0;
                    if (it2->second < mClassAvgDF[it2->first])
                        factor = pow(1.0*it2->second/mClassAvgDF[it2->first], alpha);
                    
                    weight = factor * pow(1.0*it2->second/it1->second, 1.5);
                    if (weight >= threshold) {
                        fprintf(output,"%s\t%s\t%f\t%d\t%d\t%d\t%f\t%f\n",it2->first.c_str(),it1->first.c_str(),weight,Cj[it2->first],it1->second,it2->second,1.0*it2->second/it1->second,factor);
                    }
                }
            }
        }
    }
private:
    float alpha;
    float min_count_in_class;
    float min_count;
    float threshold;
    map<string, unsigned int> DFi; // N(w)
    map<string, map<string, unsigned int> > DFij; // N(c|w)
    map<string, map<string, unsigned int> > mClassWordDF; // N(w|c)
    map<string, unsigned int> Cj;  // Pr(C)
    map<string, float>    mClassAvgDF; // 类Cj的平均词频 
};



int main() 
{
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    
    NaiveBayesianClassifier nbc;
    nbc.train(stdin, stdout);
    return 0;
}

