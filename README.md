神经网络与深度学习课程Project

组员：蒋骐泽，龚绩阳

使用fastNLP实现文章'Yoon Kim. 2014. Convolution Neural Networks for Sentence Classification.'中CNN模型并在文章提到的7个数据集进行测试
改造fastNLP.models.CNNText模型来避免运行时错误；并添加GLoVe的预训练词向量，实现文中前三种CNN模型。
结果位于result-*.txt中。其中CNN-rand结果均与文中结果误差小于2%。其余两个模型可能由于词向量的训练语料库不同所以无法达到文中的结果。
