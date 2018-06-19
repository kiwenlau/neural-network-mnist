# -*- coding: utf-8 -*-

import mnist_loader
import network

# 导入数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


'''
初始化一个3层的神经网络
- 第1层为输入层，有784个神经元, 分别代表图片的28*28=784个像素;
- 第2层为隐藏层，有30个神经元;
- 第4层为输出层，有10个神经元, 分别代表10个数字0~9;
'''
net = network.Network([784, 30, 10])


'''
采用随机梯度下降算法训练神经网络
- epochs为30，表示进行30次迭代计算;
- batch size为10, 表示每次选取10个样本进行训练;
- learning rate为3.0;
'''
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
