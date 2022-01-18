import gc
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
def train_model(train_data, test_data, y, folds):
    validation_prediction = np.zeros(train_data.shape[0])
    test_prediction = np.zeros(test_data.shape[0])
    # 剔除掉['loan_id', 'user_id', 'isDefault']这三个特征
    features = [f for f in train_data.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    # (train_index, validation_index)为索引
    for n_fold, (train_index, validation_index) in enumerate(folds.split(train_data)):
        Xtrain, Ytrain = train_data[features].iloc[train_index], y.iloc[train_index]
        XValidation, YValidation = train_data[features].iloc[validation_index], y.iloc[validation_index]
        clf = LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.08,
            num_leaves=2 ** 5,
            colsample_bytree=.65,
            subsample=.9,
            max_depth=5,
            reg_alpha=.3,
            reg_lambda=.3,
            min_split_gain=.01,
            min_child_weight=2,
            verbose=-1,
            silent=True
        )
        clf.fit(Xtrain, Ytrain,eval_set=[(XValidation, YValidation), (Xtrain, Ytrain)]
                , eval_metric='auc', verbose=100, early_stopping_rounds=40)
        # 这一轮的将验证集预测为1的概率值，经过K折验证后，train_data的每一个样本都会有预测为1的概率值，用来绘制ROC曲线，求AUC的值
        validation_prediction[validation_index] = clf.predict_proba(XValidation, num_iteration=clf.best_iteration_)[:, 1]
        # 将test_data[features]预测为1的概率（一共folds.n_splits轮预测值）的平均值。
        test_prediction += clf.predict_proba(test_data[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(YValidation, validation_prediction[validation_index])))
        del clf, Xtrain, Ytrain, XValidation, YValidation
        gc.collect()
    # 交叉验证之后，此时validation_prediction的所有 样本index均获得了值
    print('Full AUC score %.6f' % roc_auc_score(y, validation_prediction))
    result = pd.DataFrame()
    result['loan_id'] = test_data['loan_id']
    result['isDefault_prediction'] = test_prediction
    '''
    validation_prediction:将训练集的每一个样本预测为1的概率值
    test_data['isDefault']:每一次K折验证都会对测试集进行预测，这里取的是每一折预测值得平均值
    '''
    return validation_prediction, result