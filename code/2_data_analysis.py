# 导入库
import copy
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# 库设置
sns.set_style("darkgrid")
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    """---------------------------------------准备---------------------------------------"""
    # 导入完整数据
    df = pd.read_csv('../data/cleaned_google_play_store.csv')

    # 导入数值数据
    numerical_features = ['Rating', 'Reviews', 'Size', 'Installs', 'Price',
                          'Last Updated Till Now', 'Android Ver', 'Name Length',
                          'Data Flow', 'Sale Volume', 'Review Install Ratio']
    numerical_data = pd.read_csv('../data/cleaned_google_play_store.csv', usecols=numerical_features)

    # 导入机器学习格式数据
    ml_features = ['Reviews', 'Size', 'Installs', 'Price',
                   'Last Updated Till Now', 'Android Ver', 'Name Length',
                   'Data Flow', 'Sale Volume', 'Review Install Ratio']
    x = pd.read_csv('../data/cleaned_google_play_store.csv', usecols=ml_features)
    y = pd.read_csv('../data/cleaned_google_play_store.csv', usecols=['Rating', ])
    x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.3, random_state=6)

    """---------------------------------------单特征分析---------------------------------------"""
    # 单特征分布情况柱状图 (e.g. Category; Content Rating)
    plt.figure(figsize=(15, 19))
    fig = sns.countplot(x=df['Category'])
    fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
    plt.savefig('../image/category_bar.png', dpi=170)

    plt.figure(figsize=(15, 19))
    fig = sns.countplot(x=df['Content Rating'])
    fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
    plt.savefig('../image/content_rating_bar.png', dpi=170)

    # 单特征分布情况密度图 (e.g. Rating; Android Ver; Last Updated Till Now; Name Length)
    plt.figure()
    fig = sns.distplot(a=df['Rating'].values, kde=True, bins=50, color='purple', axlabel='Rating')
    plt.legend(["($\mu=${0:.2g}, $\sigma=${1:.2f})".format(np.average(df['Rating'].values),
                                                           np.std(df['Rating'].values))])
    plt.savefig('../image/rating_distribution.png', dpi=170)

    plt.figure()
    fig = sns.distplot(a=df['Last Updated Till Now'].values, kde=False, bins=50, color='purple',
                       axlabel='Last Updated Till Now')
    plt.legend(["($\mu=${0:.2g}, $\sigma=${1:.2f})".format(np.average(df['Last Updated Till Now'].values),
                                                           np.std(df['Last Updated Till Now'].values))])
    plt.savefig('../image/last_updated_till_now_distribution.png', dpi=170)

    plt.figure()
    fig = sns.distplot(a=df['Android Ver'].values, kde=True, bins=50, color='purple', axlabel='Android Ver')
    plt.legend(["($\mu=${0:.2g}, $\sigma=${1:.2f})".format(np.average(df['Android Ver'].values),
                                                           np.std(df['Android Ver'].values))])
    plt.savefig('../image/android_ver_distribution.png', dpi=170)

    plt.figure()
    fig = sns.distplot(a=df['Name Length'].values, kde=True, bins=50, color='purple', axlabel='Name Length')
    plt.legend(["($\mu=${0:.2g}, $\sigma=${1:.2f})".format(np.average(df['Name Length'].values),
                                                           np.std(df['Name Length'].values))])
    plt.savefig('../image/name_length_distribution.png', dpi=170)

    # Mean-Decreased Impurity(MDI) 数值特征重要性直方图
    rf_mdi = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=6)
    rf_mdi.fit(x, y)
    feature_importance_list = list(map(lambda x: round(x, 3), rf_mdi.feature_importances_))
    mdi_feature_importance_data_frame = pd.DataFrame()
    mdi_feature_importance_data_frame['feature_name'] = pd.Series(ml_features)
    mdi_feature_importance_data_frame['mdi_feature_importance'] = pd.Series(feature_importance_list)
    plt.figure(figsize=(15, 7))
    fig = sns.barplot(x='mdi_feature_importance', y='feature_name', data=mdi_feature_importance_data_frame)
    plt.savefig('../image/mdi_feature_importance.png', dpi=170)

    # Mean-Decreased Accuracy(MDA) 数值特征重要性直方图
    rf_mda = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=6)
    rf_mda.fit(x_train, y_train)
    original_r2_score = r2_score(y_test, rf_mda.predict(x_test))
    mda_feature_importance_list = []
    for (index, feature) in enumerate(ml_features):
        x_feature_permutation_test = copy.deepcopy(x_test)
        np.random.shuffle(x_feature_permutation_test[:, index])
        permutation_r2_score = r2_score(y_test, rf_mda.predict(x_feature_permutation_test))
        mda = (original_r2_score - permutation_r2_score) / original_r2_score
        mda_feature_importance_list.append(mda)
    mda_feature_importance_data_frame = pd.DataFrame()
    mda_feature_importance_data_frame['feature_name'] = pd.Series(ml_features)
    mda_feature_importance_data_frame['mda_feature_importance'] = pd.Series(mda_feature_importance_list)
    plt.figure(figsize=(15, 7))
    fig = sns.barplot(x='mda_feature_importance', y='feature_name', data=mda_feature_importance_data_frame)
    plt.savefig('../image/mda_feature_importance.png', dpi=170)

    """---------------------------------------双特征分析---------------------------------------"""
    # 相关性矩阵热力图 (只有数值型特征参与运算)
    correlations = numerical_data.corr()
    plt.figure(figsize=(19, 19))
    sns.heatmap(correlations, xticklabels=correlations.columns,
                yticklabels=correlations.columns, cmap='RdYlGn',
                center=0, annot=True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('../image/correlation.png', dpi=170)

    # 全数值特征散点图 (只有数值型特征参与运算)
    plt.figure(figsize=(40, 40))
    fig = sns.pairplot(numerical_data)
    plt.savefig('../image/pairwise_scatter.png', dpi=170)

    # 双特征散点图与线性回归线 (e.g. 数值单特征 vs Rating)
    plt.figure()
    fig = sns.jointplot(x='Reviews', y='Rating', data=df, color='purple', kind='reg')
    plt.savefig('../image/reviews_rating.png', dpi=170)

    plt.figure()
    fig = sns.jointplot(x='Size', y='Rating', data=df, color='purple', kind='reg')
    plt.savefig('../image/size_rating.png', dpi=170)

    plt.figure()
    fig = sns.jointplot(x='Installs', y='Rating', data=df, color='purple', kind='reg')
    plt.savefig('../image/installs_rating.png', dpi=170)

    plt.figure()
    fig = sns.jointplot(x='Last Updated Till Now', y='Rating', data=df, color='purple', kind='reg')
    plt.savefig('../image/lutn_rating.png', dpi=170)

    plt.figure()
    fig = sns.jointplot(x='Android Ver', y='Rating', data=df, color='purple', kind='reg')
    plt.savefig('../image/android_rating.png', dpi=170)

    plt.figure()
    fig = sns.jointplot(x='Name Length', y='Rating', data=df, color='purple', kind='reg')
    plt.savefig('../image/name_length_rating.png', dpi=170)

    plt.figure()
    fig = sns.jointplot(x='Data Flow', y='Rating', data=df, color='purple', kind='reg')
    plt.savefig('../image/dataflow_rating.png', dpi=170)

    plt.figure()
    fig = sns.jointplot(x='Sale Volume', y='Rating', data=df, color='purple', kind='reg')
    plt.savefig('../image/sale_volume_rating.png', dpi=170)

    plt.figure()
    fig = sns.jointplot(x='Review Install Ratio', y='Rating', data=df, color='purple', kind='reg')
    plt.savefig('../image/rir_rating.png', dpi=170)

    # 双特征散点图与线性回归线 (相关性绝对值高的双特征 e.g. Reviews vs Installs, LUTN vs Android Ver)
    plt.figure()
    fig = sns.jointplot(x='Last Updated Till Now', y='Android Ver', data=df, color='purple', kind='reg')
    plt.savefig('../image/lutn_av_.png', dpi=170)

    plt.figure()
    fig = sns.jointplot(x='Reviews', y='Installs', data=df, color='purple', kind='reg')
    plt.savefig('../image/reviews_installs_.png', dpi=170)

    # 类(category)内Rating分布密度图
    category_to_rating_list = {}
    for i in range(len(df)):
        if df['Category'].values[i] in category_to_rating_list:
            category_to_rating_list[df['Category'].values[i]].append(df['Rating'].values[i])
        else:
            category_to_rating_list[df['Category'].values[i]] = [df['Rating'].values[i], ]

    color_list = ['red', 'orange', 'grey', 'green', 'blue', 'purple', 'cyan', 'chocolate', 'coral', 'gold', 'khaki',
                  'lavender', 'maroon', 'linen', 'lime', 'navy', 'olive', 'midnightblue', 'paleturquoise', 'pink',
                  'salmon', 'sienna', 'tan', 'teal', 'wheat', 'white', 'black', 'aqua', 'aquamarine', 'azure', 'beige',
                  'bisque', 'fuchsia']
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    for (index, category) in enumerate(category_to_rating_list):
        ax1 = sns.kdeplot(np.array(category_to_rating_list[category]),
                          shade=False,
                          label=category,
                          color=color_list[index])
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Density')
    plt.savefig('../image/rating_across_category.png', dpi=170)

    # 年龄层(content Rating)内Rating分布密度图
    content_to_rating_list = {}
    for i in range(len(df)):
        if df['Content Rating'].values[i] in content_to_rating_list:
            content_to_rating_list[df['Content Rating'].values[i]].append(df['Rating'].values[i])
        else:
            content_to_rating_list[df['Content Rating'].values[i]] = [df['Rating'].values[i], ]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    for (index, content_rating) in enumerate(content_to_rating_list):
        ax1 = sns.kdeplot(np.array(content_to_rating_list[content_rating]),
                          shade=False,
                          label=content_rating,
                          color=color_list[index])
        ax1.legend(loc=(index + 1))
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Density')
    plt.savefig('../image/rating_across_content.png', dpi=170)
