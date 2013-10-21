#ifndef _SIGMOID_H_
#define _SIGMOID_H_

#include <iostream>
#include <math.h>
using namespace std;

#define EXP_TABLE_SIZE 5000
#define MAX_EXP 6

/**
 * 取x在(-MAX_EXP, MAX_EXP)之间，小于等于-MAX_EXP的取0值，大于等于MAX_EXP的取1值.
 * (-MAX_EXP, MAX_EXP)之间的偏移扩大到[0,EXP_TABLE_SIZE]的整数，进行打表.
 * 
 * 参见google的word2vec.
 */
class Sigmoid
{
public:
    Sigmoid()
    {   
        for (int i = 0; i < EXP_TABLE_SIZE; i ++) 
        {   
            double x = exp((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
            expTable[i] = x / (x+1.0f);
        }   
    }   
    
    double value(double x)
    {   
        if (x <= -MAX_EXP) return 0.0f;
        if (x >= MAX_EXP)  return 1.0f;
        return expTable[(int)((x + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0))];
    }   
private:
    double expTable[EXP_TABLE_SIZE+1];
};



#endif
