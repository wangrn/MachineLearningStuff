/**
 *
 * @author : wangrn
 * 
 * 改进的朴素贝叶斯算法, 结合词在类别中的权重, 进行适当的调权.
 * 
 * 
 * Pr(c|d) 是文档d属于类c的概率. 假定文档内词是独立的, 则Pr(c|d) = Pr(c) * Σ Pr(w|c)*Pr(w|d) .
 *      Pr(w|d)是词w在文档d中的词频, 可以是各种归一化或平滑公式.
 *      Pr(w|d)=1.0
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
#include <set>
#include <vector>
#include <errno.h>
using namespace std;


//改进的朴素贝叶斯, 模型公式参见文件开头注释.
class NaiveBayesianTrainer
{
public:
    NaiveBayesianTrainer()
    {
        alpha = 0.2;
        min_count_in_class = 0.0;
        threshold = 0.0;
        min_count = 10;
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
            strtok(NULL, "\t"); //doc id

            Cj[label] ++;
            int term_weight = 1;
            bool first = true;
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
                if (first) {if(idx[0]!='0') term_weight=10;}
                else {if(idx[0]=='u') term_weight=5;}
                first = false;
                
                DFi[idx] += term_weight;
                DFij[idx][label] += term_weight;
                mClassWordDF[label][idx] += term_weight;
                term_weight = 1;
                i++;
            }
            total ++;
            fprintf(stderr, "doc = %d\n", total);
        }
        fprintf(stderr, "DFi=%d DFij=%d mClassWordDF=%d\n", DFi.size(), DFij.size(), mClassWordDF.size()); 
        for (map<string, uint32_t>::iterator it = DFi.begin(); it!=DFi.end(); it ++)
        {
            it->second += mClassWordDF.size();
            for (map<string, map<string, uint32_t> >::iterator it2 = mClassWordDF.begin();
                    it2 != mClassWordDF.end(); it2 ++)
            {
                mClassWordDF[it2->first][it->first] ++;
                DFij[it->first][it2->first] ++;
            }
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
            if (k > 0) mClassAvgDF[it->first] = pr / k; // avg DF in class
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
						//classid word weight N(classid) N_DOC_NUM N(word) N(classid,word) N(classid,word)/N(word) factor
                        fprintf(output,"%s\t%s\t%f\t%d\t%d\t%d\t%d\t%f\t%f\n",
                                it2->first.c_str(), // classid
                                it1->first.c_str(), // word
                                weight,             // Pr(classid | word)
                                Cj[it2->first],     // N(classid), the number of clsid in corpus
                                total,              // N , the doc number of corpus
                                it1->second,        // N(word), DF(word) in corpus
                                it2->second,        // N(classid, word)  DF(word) in classid
                                1.0*it2->second/it1->second, // N(classid,word)/N(word) === P(classid|word)
                                factor);            // factor
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

class NaiveBayesianPredictor
{
public:
    struct node
    {
        float weight;
        int nCls;
        int nTotalDoc;
        int nWord;
        int nClsWord;
        float pr; // nClsWord / nWord
        float factor;
    };
    NaiveBayesianPredictor()
    {
        FILE *fp = fopen("stopword.txt", "r");
        char line[4096];
        while (fgets(line, sizeof(line), fp))
        {
            line[strlen(line)] = '\0';
            stopword.insert(line);
        }
        fprintf(stderr, "Load %d stopwords\n", stopword.size());
        fclose(fp);
    }

    bool load(string file)
    {
        char line[4096];
        char *word, *cls, *tmp;
        struct node n;
        FILE *fp = fopen(file.c_str(), "r");
        while (fgets(line, sizeof(line), fp))
        {
            //classid word weight N(classid) N(word) N(classid|word) N(classid|word)/N(word) factor
            cls = strtok(line, "\t");
            word  = strtok(NULL, "\t");
            if (stopword.find(word) != stopword.end()) continue;
            tmp = strtok(NULL, "\t"); n.weight = atof(tmp);
            tmp = strtok(NULL, "\t"); n.nCls = atoi(tmp);
            tmp = strtok(NULL, "\t"); n.nTotalDoc = atoi(tmp);
            tmp = strtok(NULL, "\t"); n.nWord = atoi(tmp);
            tmp = strtok(NULL, "\t"); n.nClsWord = atoi(tmp);
            tmp = strtok(NULL, "\t"); n.pr = atof(tmp);
            tmp = strtok(NULL, "\t"); n.factor = atof(tmp);
            model[word][cls] = n;
            prC[cls] = n.nCls*1.0/n.nTotalDoc;
        }
        fclose(fp);
        defaultCls="UNKOWN";
        float max_prob = -10000000.0;
        for(map<string, float>::iterator it = prC.begin(); it!=prC.end(); it ++)
        {
            if (it->second > max_prob)
            {
                max_prob = it->second;
                defaultCls = it->first;
            }
        }
        fprintf(stderr, "load ok. pr=%d model=%d\n", prC.size(), model.size());
    }

    string predict(vector<string> &terms)
    {
#define Optimization 1
        map<string, double> result;
        for (int i = 0; i < terms.size(); i ++)
        {
            string word = terms[i];
            if (model.find(word) == model.end()) continue;
            map<string, node> data = model[word];

            for(map<string,node>::iterator it= data.begin(); it!=data.end(); it++)
            {
                if(result.find(it->first) == result.end()) result[it->first] = 1.0f;
                //if(i == 0) result[it->first] += log(it->second.pr)*10;
                //else result[it->first] += log(it->second.pr);
                if(i == 0) result[it->first] += log(it->second.weight)*10;
                else if(word.c_str()[0] == 'u') result[it->first] += log(it->second.weight)*5;
                else result[it->first] += log(it->second.weight);
            }
        }

        string label = defaultCls;
        double  max_prob = -1000000.0;
        for(map<string,double>::iterator it= result.begin(); it!=result.end(); it++)
        {
            double prob = it->second + log(prC[it->first]);
            if (prob > max_prob)
            {
                label = it->first;
                max_prob = prob;
            }
        }
#undef Optimization
        fprintf(stderr, "result=%s %lf\n", label.c_str(), max_prob);
        return label;
    }
private:
    map<string, map<string, node> > model;
    map<string, float> prC;
    string defaultCls;
    set<string> stopword;
};

int main(int argc, char*argv[]) 
{
    if (argc != 2)
    {
        fprintf(stderr, "%s train|predict\n", argv[0]);
        return -1;
    }	
    if (strcmp(argv[1], "train") == 0)
    {
        NaiveBayesianTrainer nbc;
        nbc.train(stdin, stdout);
    }
    else
    {
        NaiveBayesianPredictor predictor;
        predictor.load("nbc.model");
        char line[4096];
        char *label, *idx, *val, *tmp, *endptr;
        double value;
        while (fgets(line, sizeof(line), stdin))
        {
            int i = 0;
            label = strtok(line, "\t");
            if (label == NULL) continue;
            tmp = strtok(NULL, "\t");
            string docid = tmp;
            vector<string> terms;
            while (true)
            {
                idx = strtok(NULL, ":");
                if (idx == NULL) break;
                val = strtok(NULL, "\t");
                if (val == NULL) break;
                //errno = 0;
                //value = strtod(val, &endptr);
                //if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                //continue;
                terms.push_back(idx);
            }
            string output = predictor.predict(terms);
            fprintf(stdout, "%s\t%s\t%s\n", label, output.c_str(), docid.c_str());
        }
    }
    return 0;
}

