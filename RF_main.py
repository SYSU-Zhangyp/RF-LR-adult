from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 定义超参数的候选范围
param_grid = {
    'n_estimators': [50, 100, 200],  # 决策树的数量
    'max_depth': [None, 10, 20],  # 决策树的最大深度
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需最小样本数
    'min_samples_leaf': [1, 2, 4]  # 叶子节点最少样本数
}

def RF():
    # 加载训练集
    train_data = pd.read_csv("dataset/train_adult_processed.csv")
    pd.set_option("display.max_columns", 102)

    # 加载测试集
    test_data = pd.read_csv("dataset/test_adult_processed.csv")
    pd.set_option("display.max_columns", 102)
    
    # 获取训练集和测试集的列名交集
    common_cols = set(train_data.columns).intersection(test_data.columns)

    # 提取特征值和目标值，只使用交集部分的列
    train_feature = train_data[list(common_cols - {'income'})] 
    train_target = train_data['income']

    # 提取测试集的特征值和目标值，同样只使用交集部分的列
    test_feature = test_data[list(common_cols - {'income'})]
    test_target = test_data['income']

    x_train = train_feature
    y_train = train_target
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    x_test = test_feature
    y_test = test_target

    print("训练集：", x_train.shape, y_train.shape)
    print("验证集：", x_val.shape, y_val.shape)
    print("测试集：", x_test.shape, y_test.shape)

    # 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_val = std.transform(x_val)
    x_test = std.transform(x_test)

    # 训练数据采样
    # x_train, y_train = to_over_sample(x_train, y_train)
    # x_train, y_train = to_under_sample(x_train, y_train)

    # 测试数据采样
    # 测试集应该尽可能地反映真实世界中的数据分布，以便评估模型在实际应用场景下的性能，不需要采样。
    # x_test, y_test = to_over_sample(x_test, y_test)
    # x_test, y_test = to_under_sample(x_test, y_test)    

    # 创建一个随机森林分类器对象
    rf_classifier = RandomForestClassifier(random_state=42)

    # 创建网格搜索对象
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # 在训练集上拟合网格搜索对象
    grid_search.fit(x_train, y_train)

    # 输出最佳参数组合
    print("最佳参数组合:", grid_search.best_params_)

    # 输出最佳得分
    print("最佳得分:", grid_search.best_score_)

    # 使用最佳参数拟合最佳模型
    best_rf_classifier = grid_search.best_estimator_

    # 使用最佳模型进行预测
    predictions = best_rf_classifier.predict(x_test)

    # 绘制ROC曲线
    to_plot(best_rf_classifier, x_test, y_test)

    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, predictions)


def to_over_sample(x_train, y_train):
    # 创建 RandomOverSampler 对象
    ros = RandomOverSampler()

    # 对训练集进行上采样
    x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)
    print(len(x_train_resampled), len(y_train_resampled))

    return x_train_resampled, y_train_resampled

def to_under_sample(x_train, y_train):
    # 创建 RandomOverSampler 对象
    rus = RandomUnderSampler()

    # 对训练集进行上采样
    x_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)
    print(len(x_train_resampled), len(y_train_resampled))
    
    return x_train_resampled, y_train_resampled

def plot_confusion_matrix(y_true, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 获取类别标签
    labels = ['Negative', 'Positive']
    
    # 创建热图
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    
    # 添加标签
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j + 0.5, i + 0.5, cm[i, j], ha='center', va='center', color='black', fontsize=12)
    
    plt.show()

def to_plot(rf, x_test, y_test):
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(x_test)[:,1])
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    RF()