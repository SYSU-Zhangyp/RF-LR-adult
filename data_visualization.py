import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualization_continuous_column(df, i, size_x, size_y, tag):
    # 获取特征列的列名列表（假设特征列从第i列到第j列）
    feature_columns = df.columns[i:i+1]

    # 设置直方图的参数
    num_bins = 10
    alpha = 0.5

    # 循环遍历每个特征列，绘制数据直方图
    for feature in feature_columns:
        plt.figure(figsize=(size_x, size_y))
        
        # 根据标签列的值拆分数据集
        label_0_data = df[df['income'] == 0][feature]
        label_1_data = df[df['income'] == 1][feature]

        # 计算直方图的 bin 边界
        min_val = min(label_0_data.min(), label_1_data.min())
        max_val = max(label_0_data.max(), label_1_data.max())
        bins = np.linspace(min_val, max_val, num_bins)

        # 绘制直方图
        plt.hist([label_0_data, label_1_data], bins, alpha=alpha, align='mid', label=['≤50k', '>50k'])

        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {feature} by Label')
        plt.legend(loc='upper right')
        plt.show()

    return

if __name__ == "__main__":
    df_train_set = pd.read_csv('dataset/train_adult_processed.csv')
    df_test_set = pd.read_csv('dataset/test_adult_processed.csv')
    df = df_test_set + df_train_set

    index_age = df.columns.get_loc('age')
    index_capitalGain = df.columns.get_loc('capitalGain')
    index_capitalLoss = df.columns.get_loc('capitalLoss')
    index_hoursPerWeek = df.columns.get_loc('hoursPerWeek')
    # visualization_continuous_column(df, index_age, 6, 6, tag='age')
    # visualization_continuous_column(df, index_capitalGain, 6, 6, tag='capitalGain')
    # visualization_continuous_column(df, index_capitalLoss, 6, 6, tag='capitalLoss')
    visualization_continuous_column(df, index_hoursPerWeek, 6, 6, tag='hoursPerWeek')

