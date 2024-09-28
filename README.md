# Stochastic Activation Pruninging

原理就是，在每个RELU后面，加一个类似dropout一样的层，这个层根据每个值的权重大小来随机dropout。

每个标量的权重就是这个标量的绝对值占所有标量绝对值之和的比重。
