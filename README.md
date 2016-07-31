# DLNLP
### 如何构建LSTM的输入数据（X）和标签(label)：
B：begin  E：end  M:middle   S:sigle


例如：小明喜欢自然语言处理的问题

分词后的结果为：小B明E喜B欢E自B然E语B言E处B理E的S问B题E
>训练数据：

X = [[[w(小)][w(明)][w(喜)]...]] Y = [[B,E,B,E...]...]
最外层list代表所有的句子，第二层list代表一个句子，第三层list代表一个字的窗口比如w(明)= [小，明，喜]

dataUtils 数据预处理：
- 将word映射为index
- 把n个字的句子映射为[x1,x2,x3,...,xn]。其中x_i为list，指w_i的context.
- dataUtils也计算了[B,E,M,S]的初始概率和相互的转移概率。

