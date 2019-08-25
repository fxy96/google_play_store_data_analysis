# 导入依赖库
import time
import warnings
import datetime
import numpy as np
import pandas as pd


# 库设置
warnings.filterwarnings('ignore')


# 计算两天之间的天数
def between_day_count(date1: str, date2: str):
    date1 = time.strptime(date1, "%Y/%m/%d")
    date2 = time.strptime(date2, "%Y/%m/%d")
    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    return float((date2 - date1).days)


# 将英文月份转换为数字
def month2num(month: str):
    if month == 'January':
        return '1'
    elif month == 'February':
        return '2'
    elif month == 'March':
        return '3'
    elif month == 'April':
        return '4'
    elif month == 'May':
        return '5'
    elif month == 'June':
        return '6'
    elif month == 'July':
        return '7'
    elif month == 'August':
        return '8'
    elif month == 'September':
        return '9'
    elif month == 'October':
        return '10'
    elif month == 'November':
        return '11'
    elif month == 'December':
        return '12'
    else:
        raise ValueError


if __name__ == '__main__':
    """---------------------------------------数据清洗---------------------------------------"""
    # 导入数据
    df = pd.read_csv('../data/google_play_store.csv')

    # 将App评分(Rating)转为浮点数
    df['Rating'] = df['Rating'].apply(lambda x: float(x))

    # 将App评论量(Reviews)转为浮点数
    df['Reviews'] = df['Reviews'].apply(lambda x: float(x))

    # 将App大小(Size)转换为同一单位(MB)的浮点数
    df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
    df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
    df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1024 if 'k' in str(x) else x)
    df['Size'] = df['Size'].apply(lambda x: float(x))

    # 将App下载量(Installs)转为浮点数
    df['Installs'] = df['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else x)
    df['Installs'] = df['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else x)
    df['Installs'] = df['Installs'].apply(lambda x: float(x))

    # 将App价格(Price)转为浮点数
    df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else x)
    df['Price'] = df['Price'].apply(lambda x: float(x))

    # 将App最后一次更新的日期(Last Updated)转变为距今天数(浮点数)
    df['Last Updated'] = df['Last Updated'].apply(lambda x: between_day_count((x.split(',')[1] + '/' + month2num(x.split(' ')[0]) + '/' + x.split(' ')[1].split(',')[0]).replace(' ', ''), '2019/1/1'))
    df['Last Updated'] = df['Last Updated'].apply(lambda x: float(x))
    df.rename(columns={'Last Updated': 'Last Updated Till Now'}, inplace=True)

    # 将App适配的安卓版本(Android Ver)转为浮点数
    df['Android Ver'] = df['Android Ver'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
    df['Android Ver'] = df['Android Ver'].apply(lambda x: str(x).replace(' and up', '') if ' and up' in str(x) else x)
    df['Android Ver'] = df['Android Ver'].apply(lambda x: str(x).replace('.', '')[0] + '.' + str(x).replace('.', '')[1:] if '.' in str(x) else x)
    df['Android Ver'] = df['Android Ver'].apply(lambda x: x[0: x.find(' ')] if '-' in str(x) else x)
    df['Android Ver'] = df['Android Ver'].apply(lambda x: float(x))

    # 对评分(Rating)做缺失值填充(对缺失值用同类App评分的均值来填充)
    category_rating = {}
    for i in range(len(df['App'])):
        if np.isnan(df['Rating'].values[i]):
            continue
        elif df['Category'].values[i] in category_rating:
            category_rating[df['Category'].values[i]][0] += 1
            category_rating[df['Category'].values[i]][1] += df['Rating'].values[i]
        else:
            category_rating[df['Category'].values[i]] = [1, df['Rating'].values[i]]

    for i in range(len(df['App'])):
        if np.isnan(df['Rating'].values[i]):
            category_average_rating = category_rating[df['Category'].values[i]][1] / \
                                      category_rating[df['Category'].values[i]][0]
            df['Rating'][i] = category_average_rating
        else:
            continue

    # 对大小(Size)做缺失值填充(对缺失值用同类App大小的均值来填充)
    category_size = {}
    for i in range(len(df['App'])):
        if np.isnan(df['Size'].values[i]):
            continue
        elif df['Category'].values[i] in category_size:
            category_size[df['Category'].values[i]][0] += 1
            category_size[df['Category'].values[i]][1] += df['Size'].values[i]
        else:
            category_size[df['Category'].values[i]] = [1, df['Size'].values[i]]

    for i in range(len(df['App'])):
        if np.isnan(df['Size'].values[i]):
            category_average_size = category_size[df['Category'].values[i]][1] / \
                                    category_size[df['Category'].values[i]][0]
            df['Size'][i] = category_average_size
        else:
            continue

    # 对安卓版本(Android Ver)做缺失值填充(对缺失值用同类App安卓版本的均值来填充)
    category_ver = {}
    for i in range(len(df['App'])):
        if np.isnan(df['Android Ver'].values[i]):
            continue
        elif df['Category'].values[i] in category_ver:
            category_ver[df['Category'].values[i]][0] += 1
            category_ver[df['Category'].values[i]][1] += df['Android Ver'].values[i]
        else:
            category_ver[df['Category'].values[i]] = [1, df['Android Ver'].values[i]]

    for i in range(len(df['App'])):
        if np.isnan(df['Android Ver'].values[i]):
            category_average_version = category_ver[df['Category'].values[i]][1] / \
                                       category_ver[df['Category'].values[i]][0]
            df['Android Ver'][i] = category_average_version
        else:
            continue

    """---------------------------------------特征工程---------------------------------------"""
    # 名字的字符个数
    name_length_list = [float(len(app_name)) for app_name in df['App']]
    df['Name Length'] = pd.Series(name_length_list)

    # 总下载流量
    data_flow_list = [float(df['Size'].values[i] * df['Installs'].values[i]) for i in range(len(df))]
    df['Data Flow'] = pd.Series(data_flow_list)

    # 总销售额
    sale_volume_list = [float(df['Price'].values[i] * df['Installs'].values[i]) for i in range(len(df))]
    df['Sale Volume'] = pd.Series(sale_volume_list)

    # 评论下载比
    review_install_ratio_list = [float(df['Reviews'].values[i] / df['Installs'].values[i] if df['Installs'].values[i] > 1e-6 else 0) for i in range(len(df))]
    df['Review Install Ratio'] = pd.Series(review_install_ratio_list)
    
    # 保存清洗好的数据到磁盘上
    df.to_csv('../data/cleaned_google_play_store.csv', index=False)
