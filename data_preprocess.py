import pandas as pd

def preprocess(in_path, out_path):
    """  
    预处理函数，用于读取并处理CSV文件。  

    参数:  
    in_path: 输入文件的路径  
    out_path: 输出文件的路径  

    返回值:  
    无返回值，处理后的DataFrame将直接保存到out_path指定的路径  
    """ 
    # 读取路径下的文件，转为 DataFrame 对象
    df = pd.read_csv(in_path)

    # 这里考虑删去 fnlwgt 数据和 educationNum 数据(fnlwgt列用处不大，educationNum与education类似，数据重复删去)
    df.drop(['fnlwgt', 'educationNum'], axis=1, inplace=True) 
    # print(df.columns)

    # 对数据进行初步的清洗
    # 去除重复行，重复的数据是无效的
    df.drop_duplicates(inplace=True)

    # 去除空行，空行容易出现错误 
    df.dropna(inplace=True) 
    # print(df.shape[0])

    # 查找异常值, 避免与正则表达式的?冲突需要转义
    # 需要查找异常值的列名列表 
    new_columns = ['workclass', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry', 'income']
    for col in new_columns:
        df = df[~df[col].str.contains(r'\?', regex=True)]
    print(df.shape[0])
    print(df.head)

    # 对连续特征进行范围编码 
    continuous_column = ['age', 'capitalGain',  'capitalLoss', 'hoursPerWeek']
    age_bins = [0, 23, 28, 33, 38, 43, 53, 63, 100]
    df['age'] = pd.cut(df['age'], age_bins, labels=False)
    capGain_bins = [-1, 100, 1000, 5000, 10000, 30000, 50000, 100000, 150000]
    df['capitalGain'] = pd.cut(df['capitalGain'], capGain_bins, labels=False)
    capLossbins = [-1, 1, 1500, 2000, 4000, 6000, 10000]
    df['capitalLoss'] = pd.cut(df['capitalLoss'], capLossbins, labels=False)
    hourbins = [0, 10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 100]
    df['hoursPerWeek'] = pd.cut(df['hoursPerWeek'], hourbins, labels=False)

    # 对离散特征进行独热编码  
    # categorical_columns = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry']
    categorical_columns = ['workclass', 'education', 'maritalStatus', 'occupation', 'nativeCountry']
    df = pd.get_dummies(df, columns=categorical_columns)

    # 编码 income 列
    income_mapping = {' <=50K': 0, ' <=50K.': 0,' >50K': 1,' >50K.': 1}
    df['income'] = df['income'].map(income_mapping)

    # 删除不需要的特征列
    columns_to_drop = ['relationship', 'race', 'sex']  
    df.drop(columns=columns_to_drop, inplace=True)

    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
        'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
    df_train_set = pd.read_csv('dataset/adult_train.txt', names=columns)
    df_test_set = pd.read_csv('dataset/adult_test.txt', names=columns, skiprows=1)

    df_train_set.to_csv('dataset/train_adult.csv', index=False)
    df_test_set.to_csv('dataset/test_adult.csv', index=False)
    preprocess('dataset/train_adult.csv','dataset/train_adult_processed.csv')
    preprocess('dataset/test_adult.csv','dataset/test_adult_processed.csv')
