# 导入工具
import copy
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import OrderedDict
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# 库设置
sns.set_style("darkgrid")
warnings.filterwarnings("ignore")


# Stacking集成学习模型
class MetaLearner(object):
    """构造函数"""
    def __init__(self, hyper_parameters_dict):
        # 模型超参数
        """
        'lasso_alpha': lasso回归的1-范数惩罚大小
        'base_ridge_alpha': 基学习器ridge回归的2-范数惩罚大小
        'svr_c': 支持向量机回归对软间隔的惩罚大小
        'svr_epsilon': 支持向量机回归对误差的最大允许
        'rf_tree_depth': 随机森林中单棵树的最大树深
        'gbt_tree_depth': 梯度提升树中单棵树的最大树深
        'gbt_lr': 梯度提升树的学习率
        'meta_ridge_alpha': 顶层学习器ridge回归的2-范数惩罚大小
        """
        self.hyper_parameters = copy.deepcopy(hyper_parameters_dict)

        # 基学习器
        self.base_learners = OrderedDict()
        self.base_learners['lasso'] = Lasso(alpha=self.hyper_parameters['lasso_alpha'])
        self.base_learners['ridge'] = Ridge(alpha=self.hyper_parameters['base_ridge_alpha'])
        self.base_learners['svm'] = SVR(C=self.hyper_parameters['svr_c'],
                                        epsilon=self.hyper_parameters['svr_epsilon'])
        self.base_learners['rf'] = RandomForestRegressor(max_depth=int(self.hyper_parameters['rf_tree_depth']),
                                                         n_estimators=500,
                                                         random_state=1,
                                                         n_jobs=-1)
        self.base_learners['gbt'] = GradientBoostingRegressor(max_depth=int(self.hyper_parameters['gbt_tree_depth']),
                                                              learning_rate=self.hyper_parameters['gbt_lr'],
                                                              loss='huber',
                                                              n_estimators=300)

        # 顶层学习器
        self.meta_learner = Ridge(alpha=self.hyper_parameters['meta_ridge_alpha'])

    """训练"""
    def train(self, x, y, verbose=False):
        # 训练基学习器
        for base_learner in self.base_learners:
            if verbose:
                print('Train base learner: ' + str(base_learner))
            self.base_learners[base_learner].fit(x, y)

        # 构造供顶层学习器训练的训练集
        x_meta = []
        for base_learner in self.base_learners:
            x_meta.append(np.reshape(self.base_learners[base_learner].predict(x), newshape=len(x)))
        x_meta = np.transpose(np.array(x_meta))

        # 训练顶层学习器
        if verbose:
            print('Train meta learner')
        self.meta_learner.fit(x_meta, y)

    """预测"""
    def predict(self, x):
        # 基模型进行预测
        x_meta = []
        for base_learner in self.base_learners:
            x_meta.append(np.reshape(self.base_learners[base_learner].predict(x), newshape=len(x)))
        x_meta = np.transpose(np.array(x_meta))

        # 顶层模型进行预测
        y_predict = np.reshape(self.meta_learner.predict(x_meta), newshape=len(x))
        return y_predict

    """评价"""
    def evaluate(self, x, y):
        # 进行预测
        y_predict = self.predict(x)

        # 评价
        mae = metrics.mean_absolute_error(y_true=y, y_pred=y_predict)
        rmse = np.sqrt(metrics.mean_squared_error(y_true=y, y_pred=y_predict))
        r2 = metrics.r2_score(y_true=y, y_pred=y_predict)

        # 显示结果
        print('MAE: %f' % mae)
        print('RMSE: %f' % rmse)
        print('R2: %f' % r2)
        return mae, rmse, r2


# 导入机器学习数据
def load_ml_data():
    # 导入原始数据
    ml_features = ['Reviews', 'Size', 'Installs', 'Price',
                   'Last Updated Till Now', 'Android Ver', 'Name Length',
                   'Data Flow', 'Sale Volume', 'Review Install Ratio']
    ml_label = ['Rating', ]
    x_csv = pd.read_csv('../data/cleaned_google_play_store.csv', usecols=ml_features)
    y_csv = pd.read_csv('../data/cleaned_google_play_store.csv', usecols=ml_label)

    # 最大最小值标准化
    scaler = MinMaxScaler()
    scaler.fit(x_csv.values)
    x_scaled = np.array(scaler.transform(x_csv.values))

    # 训练-测试集分割
    train_x, test_x, train_y, test_y = train_test_split(x_scaled, y_csv.values, test_size=0.20, random_state=666666)
    del x_csv, y_csv
    return train_x, test_x, train_y, test_y


# 贝叶斯超参数优化
def bayesian_hyper_parameter_optimization():
    # 在磁盘上创建CSV文件以记录贝叶斯超参数优化的过程
    bo = open('../data/bo_record.csv', mode='w')
    bo.write(
        'lasso_alpha,base_ridge_alpha,svr_c,svr_epsilon,rf_tree_depth,gbt_tree_depth,gbt_lr,meta_ridge_alpha,nmae\n')
    bo.close()

    # 指定超参数范围
    bounds = {'lasso_alpha': (0.1, 5), 'base_ridge_alpha': (0.1, 5), 'svr_c': (0.1, 5),
              'svr_epsilon': (0.01, 0.5), 'rf_tree_depth': (3, 20), 'gbt_tree_depth': (3, 20),
              'gbt_lr': (0.01, 0.5), 'meta_ridge_alpha': (0.1, 5)}

    # 定义贝叶斯优化器
    optimizer = BayesianOptimization(f=objective_function,
                                     pbounds=bounds,
                                     random_state=58791, )

    # 贝叶斯优化
    optimizer.maximize(init_points=20, n_iter=500)
    return optimizer.max


# 贝叶斯超参数优化的目标函数
def objective_function(lasso_alpha, base_ridge_alpha, svr_c, svr_epsilon,
                       rf_tree_depth, gbt_tree_depth, gbt_lr, meta_ridge_alpha):
    # 构造智能体
    bo_hpd = {'lasso_alpha': lasso_alpha, 'base_ridge_alpha': base_ridge_alpha, 'svr_c': svr_c,
              'svr_epsilon': svr_epsilon, 'rf_tree_depth': rf_tree_depth, 'gbt_tree_depth': gbt_tree_depth,
              'gbt_lr': gbt_lr, 'meta_ridge_alpha': meta_ridge_alpha}
    meta_learner = MetaLearner(hyper_parameters_dict=bo_hpd)

    # 导入数据
    x, _, y, _ = load_ml_data()
    train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=0.30, random_state=666666)

    # 训练
    meta_learner.train(x=train_x, y=train_y)

    # 测试
    bo_nmae = - metrics.mean_absolute_error(y_true=validation_y, y_pred=meta_learner.predict(validation_x))

    # 在磁盘上记录本次搜索结果
    bo = open('../data/bo_record.csv', mode='a')
    bo.write(str(lasso_alpha) + ',')
    bo.write(str(base_ridge_alpha) + ',')
    bo.write(str(svr_c) + ',')
    bo.write(str(svr_epsilon) + ',')
    bo.write(str(rf_tree_depth) + ',')
    bo.write(str(gbt_tree_depth) + ',')
    bo.write(str(gbt_lr) + ',')
    bo.write(str(meta_ridge_alpha) + ',')
    bo.write(str(bo_nmae) + '\n')
    bo.close()

    # 返回负平均绝对误差供贝叶斯优化使用
    return bo_nmae


# 绘制贝叶斯优化过程曲线
def plot_bayesian_optimization_process():
    nmae_curve = pd.read_csv('../data/bo_record.csv', usecols=['nmae', ])
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1 = sns.lineplot(data=nmae_curve['nmae'].values,
                       color='darkblue',
                       label='NMAE')
    ax1 = sns.lineplot(data=np.array([max(nmae_curve['nmae'].values[0: i + 1]) for i in range(len(nmae_curve['nmae']))]),
                       color='red',
                       label='Best_NMAE')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('NMAE')
    plt.savefig('../image/bayesian_optimization.png', dpi=170)


if __name__ == '__main__':
    "------------------------------------贝叶斯超参数调优------------------------------------"
    # 进行贝叶斯超参数优化
    # bo_result = bayesian_hyper_parameter_optimization()

    # 绘制贝叶斯优化过程曲线
    plot_bayesian_optimization_process()

    "------------------------------------训练机器学习模型------------------------------------"
    # 导入数据
    x_train, x_test, y_train, y_test = load_ml_data()

    # 构造Stacking集成学习模型
    # model = MetaLearner(hyper_parameters_dict=bo_result['params'])
    model = MetaLearner(hyper_parameters_dict={'lasso_alpha': 4.54575588123347, 'base_ridge_alpha': 4.84532662518666,
                                               'svr_c': 4.50590520002009, 'svr_epsilon': 0.184765912407519,
                                               'rf_tree_depth': 3, 'gbt_tree_depth': 3,
                                               'gbt_lr': 0.0191699607174971, 'meta_ridge_alpha': 4.81264355750884})

    # 训练
    model.train(x=x_train, y=y_train, verbose=True)

    "------------------------------------模型表现测试------------------------------------"
    # 测试Stacking集成模型
    _, _, r2_stacking = model.evaluate(x=x_test, y=y_test)

    # 测试Lasso子模型
    r2_lasso = metrics.r2_score(y_true=y_test,
                                y_pred=np.reshape(a=model.base_learners['lasso'].predict(x_test),
                                                  newshape=len(y_test)))

    # 测试Ridge子模型
    r2_ridge = metrics.r2_score(y_true=y_test,
                                y_pred=np.reshape(a=model.base_learners['ridge'].predict(x_test),
                                                  newshape=len(y_test)))

    # 测试SVM子模型
    r2_svm = metrics.r2_score(y_true=y_test,
                              y_pred=np.reshape(a=model.base_learners['svm'].predict(x_test),
                                                newshape=len(y_test)))

    # 测试RF子模型
    r2_rf = metrics.r2_score(y_true=y_test,
                             y_pred=np.reshape(a=model.base_learners['rf'].predict(x_test),
                                               newshape=len(y_test)))

    # 测试GBT子模型
    r2_gbt = metrics.r2_score(y_true=y_test,
                              y_pred=np.reshape(a=model.base_learners['gbt'].predict(x_test),
                                                newshape=len(y_test)))

    # 可视化不同模型的R2
    model_performance = pd.DataFrame()
    model_performance['model_name'] = pd.Series(['stacking', 'lasso', 'ridge', 'svr', 'rf', 'gbt'])
    model_performance['model_r2'] = pd.Series([r2_stacking, r2_lasso, r2_ridge, r2_svm, r2_rf, r2_gbt])
    plt.figure(figsize=(8, 4))
    fig = sns.barplot(x='model_r2',
                      y='model_name',
                      data=model_performance,
                      palette=['darkblue', 'yellow', 'red', 'green', 'purple', 'chocolate'])
    plt.savefig('../image/model_performance.png', dpi=170)

    # Stacking内部权重, 即顶层学习器Ridge回归的回归系数
    model_importance = pd.DataFrame()
    model_importance['model_name'] = pd.Series(['lasso', 'ridge', 'svr', 'rf', 'gbt'])
    model_importance['model_weight'] = pd.Series(model.meta_learner.coef_[0])
    plt.figure(figsize=(8, 4))
    fig = sns.barplot(x='model_weight',
                      y='model_name',
                      data=model_importance,
                      palette=['yellow', 'red', 'green', 'purple', 'chocolate'])
    plt.savefig('../image/model_importance.png', dpi=170)

    # 测试做特征工程前后, 机器学习算法表现对比
    ml_no_engineer_features = ['Reviews', 'Size', 'Installs', 'Price', 'Last Updated Till Now', 'Android Ver']
    ml_no_engineer_label = ['Rating', ]
    x_no_engineer_csv = pd.read_csv('../data/cleaned_google_play_store.csv', usecols=ml_no_engineer_features)
    y_no_engineer_csv = pd.read_csv('../data/cleaned_google_play_store.csv', usecols=ml_no_engineer_label)
    scaler = MinMaxScaler()
    scaler.fit(x_no_engineer_csv.values)
    x_no_engineer_scaled = np.array(scaler.transform(x_no_engineer_csv.values))
    train_no_engineer_x, test_no_engineer_x, train_no_engineer_y, test_no_engineer_y = train_test_split(x_no_engineer_scaled,
                                                                                                        y_no_engineer_csv.values,
                                                                                                        test_size=0.20,
                                                                                                        random_state=666666)
    model_no_engineer_feature = MetaLearner(hyper_parameters_dict={'lasso_alpha': 4.54575588123347, 'base_ridge_alpha': 4.84532662518666,
                                                                   'svr_c': 4.50590520002009, 'svr_epsilon': 0.184765912407519,
                                                                   'rf_tree_depth': 3, 'gbt_tree_depth': 3,
                                                                   'gbt_lr': 0.0191699607174971, 'meta_ridge_alpha': 4.81264355750884})

    model_no_engineer_feature.train(x=train_no_engineer_x, y=train_no_engineer_y)
    _, _, r2_no_engineer_feature_stacking = model_no_engineer_feature.evaluate(x=test_no_engineer_x,
                                                                               y=test_no_engineer_y)
    feature_engineering_comparison = pd.DataFrame()
    feature_engineering_comparison['is_engineered'] = pd.Series(['Engineered', 'Non-Engineered'])
    feature_engineering_comparison['R2'] = pd.Series([r2_stacking, r2_no_engineer_feature_stacking])
    plt.figure(figsize=(5, 4))
    fig = sns.barplot(x='is_engineered',
                      y='R2',
                      data=feature_engineering_comparison,
                      palette=['darkblue', 'chocolate'])
    plt.savefig('../image/feature_engineer_comparison.png', dpi=170)
