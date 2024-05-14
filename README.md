# 实验9：Adult数据集预测

本次的实验要求对 Adult 数据集进行分析， 这个任务其实本质上属于二分类问题，我们需
要使用逻辑回归和随机森林等多种模型，并结合类别不平衡处理方法、特征工程技术和超参数
调优等优化技术进行数据预测任务，即判断一个人的年收入是否超过 5 万。 使用多种的评判指
标，可以包括平均准确率(Average Accuracy)和 AUC(Area Under the ROC Curve)，以及其他可以用于评价的指标，并需要与之前的决策树模型进行对比。

## 实验环境

操作系统： Windows 11 Version 23H2

Python 版本： Python 3.11.4 ('base':conda)

## 目录结构

```shell
—— RF-LR-adult
 |—— dataset 数据集文件夹
 |—— figure  图像结果数据
 |—— data_preprocess.py 数据清洗、编码
 |—— data_visualization.py 数据可视化、特征编码调优
 |—— LR_main.py 逻辑回归模型主函数
 |—— RF_main.py 随机森林主函数
```

## 测试运行

首先运行数据清洗、编码。

```shell
python data_preprocess.py
```

必要时可以修改编码逻辑，可以基于数据可视化程序。
```shell
python data_visualization.py
```

运行逻辑回归模型。
```shell
python LR_main.py
```

运行随机森林模型。
```shell
python RF_main.py
```

## 模型调优

可以对训练集数据进行不平稳处理(过采样或欠采样)
```python
    # 训练数据采样
    # x_train, y_train = to_over_sample(x_train, y_train)
    x_train, y_train = to_under_sample(x_train, y_train)
```

可以对模型进行权重平衡
```python
    # 建立模型
    lg = LogisticRegression()
    # lg = LogisticRegression(class_weight='balanced')
```

可以调整随机森林的超参数

```python

    # 定义超参数的候选范围
    param_grid = {
        'n_estimators': [50, 100, 200],  # 决策树的数量
        'max_depth': [None, 10, 20],  # 决策树的最大深度
        'min_samples_split': [2, 5, 10],  # 内部节点再划分所需最小样本数
        'min_samples_leaf': [1, 2, 4]  # 叶子节点最少样本数
    }
```
