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

int main() 
{
    printf ("Hello world!\n");
    return 0;
}
