# EC601-Project2
## Keras
- data_format  
TensorFlow模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为（100,16,32,3），即把通道维放在了最后，这种数据组织方式称为“channels_last”.即：第0个维度是样本维，代表样本的数目，第3个维度是通道维，代表颜色通道数
- functional model API  
一种叫Sequential的模型是特殊情况，称为序贯模型，也就是单输入单输出
- tensor  
张量可以看作是向量、矩阵的自然推广，我们用张量来表示广泛的数据类型.规模最小的张量是0阶张量
- batch  
batch就是梯度下降,每次的参数更新有两种方式:Batch gradient descent批梯度下降，和stochastic gradient descent随机梯度下降.  
现在一般采用折中手段：mini-batch gradient decent，小批的梯度下降.Keras的模块中经常会出现batch_size，就是指这个  
一个batch由若干条数据构成,batch是进行网络优化的基本单位,网络参数的每一轮优化需要使用一个batch.batch中的样本是被并行处理的
- epochs  
就是训练过程中数据将被轮多少次

## Reference
- [深度学习及TensorFlow简介](http://www.infoq.com/cn/articles/introduction-of-tensorflow-part01)
- [TensorFlow（1）入门实例](http://www.jeyzhang.com/tensorflow-learning-notes.html)
- [TensorFlow（2）构建CNN模型](http://www.jeyzhang.com/tensorflow-learning-notes-2.html)
- [Keras深度学习框架(中)](https://keras-cn.readthedocs.io/en/latest/for_beginners/concepts/)
- [Keras深度学习框架(英)](https://keras.io/getting-started/sequential-model-guide/)
- [Keras之用CNN分类Mnist实例](https://blog.csdn.net/pandamax/article/details/60578077)(关联的github里有其余相关内容教学)
