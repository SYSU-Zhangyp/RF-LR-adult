import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.metrics import accuracy_score

def LR():
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
    x_train, y_train = to_under_sample(x_train, y_train)

    # 测试数据采样
    # 测试集应该尽可能地反映真实世界中的数据分布，以便评估模型在实际应用场景下的性能，不需要采样。
    # x_test, y_test = to_over_sample(x_test, y_test)
    # x_test, y_test = to_under_sample(x_test, y_test)    

    # 建立模型
    lg = LogisticRegression()
    # lg = LogisticRegression(class_weight='balanced')

    # 训练
    lg.fit(x_train, y_train)
    
    # 验证
    score_val = lg.score(x_val, y_val)
    print("在验证集上的得分：", score_val)
    
    # 测试
    score_test = lg.score(x_test, y_test)
    print("在测试集上的得分：", score_test)
    
    # 预测
    predict = lg.predict(x_test)
    fpr, tpr, thresholds = to_plot(lg, predict, x_test, y_test)

    set_thresholds = 0.69
    # 调整阈值重新预测
    predict = re_predict(lg, set_thresholds, x_test, y_test)
    # to_plot(lg, predict, x_test, y_test)

    plot_threshold_distribution(lg, x_test, y_test)

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

def to_plot(lg, predict, x_test, y_test):
    print(predict)
    
    # 打印召回率，F1
    print(classification_report(y_test, predict, labels=[0, 1], target_names=["正样本", "负样本"]))
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, lg.predict_proba(x_test)[:,1])
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

    # 计算 Precision-Recall 曲线
    precision, recall, thresholds = precision_recall_curve(y_test, lg.predict_proba(x_test)[:, 1])

    # 绘制 Precision-Recall 曲线
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()    

    return fpr, tpr, thresholds

def re_predict(lg, threshold, x_test, y_test):
    # 根据最佳阈值重新预测
    predict_adjusted = (lg.predict_proba(x_test)[:, 1] > threshold).astype(int)

    # 计算准确率
    accuracy = accuracy_score(y_test, predict_adjusted)
    print("调整后模型的准确率:", accuracy)
    return predict_adjusted

def plot_threshold_distribution(lg, x_test, y_test):
    # 获取模型在测试集上的预测概率
    probabilities = lg.predict_proba(x_test)[:, 1]

    # 均匀取样 100 个点作为阈值
    thresholds = np.linspace(0, 1, 100)

    # 初始化正类和负类的数量列表
    positive_counts = []
    negative_counts = []

    # 遍历阈值，统计每个阈值下正类和负类的数量
    for threshold in thresholds:
        # 根据预测概率和阈值将样本划分为正类和负类
        predicted_positive = probabilities < threshold
        predicted_negative = probabilities >= threshold

        # 统计正类和负类的数量
        positive_count = np.sum(predicted_positive[y_test == 1])
        negative_count = np.sum(predicted_negative[y_test == 0])

        # 添加到列表中
        positive_counts.append(positive_count)
        negative_counts.append(negative_count)

    # 绘制两条曲线
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, positive_counts, label='>=50k')
    plt.plot(thresholds, negative_counts, label='<50k')
    plt.xlabel('Threshold')
    plt.ylabel('Sample Count')
    plt.title('Threshold Distribution for Positive and Negative Classes')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    LR()

