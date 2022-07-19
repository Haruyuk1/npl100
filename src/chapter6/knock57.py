from asyncore import write
import os
import pickle
from io import TextIOWrapper
import string
import re

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def main(f: TextIOWrapper):
    train_path = 'data/train.txt'
    valid_path = 'data/valid.txt'
    test_path = 'data/test.txt'

    headers = ["ID","TITLE","URL","PUBLISHER","CATEGORY","STORY","HOSTNAME","TIMESTAMP"]
    train = pd.read_table(train_path, header=None, names=headers, index_col='ID')
    valid = pd.read_table(valid_path, header=None, names=headers, index_col='ID')
    test = pd.read_table(test_path, header=None, names=headers, index_col='ID')

    def preprocess(text: str):
        table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        text = text.translate(table)  # 記号をスペースに置換
        text = text.lower()  # 小文字化
        text = re.sub('[0-9]+', '0', text)  # 数字列を0に置換
        return text

    train['TITLE'] = train['TITLE'].map(lambda x: preprocess(x)) 
    valid['TITLE'] = valid['TITLE'].map(lambda x: preprocess(x))
    test['TITLE'] = test['TITLE'].map(lambda x: preprocess(x))

    # TfidfVectorizer
    vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))

    train_valid = pd.concat([train, valid])
    vec_tfidf = vec_tfidf.fit(train_valid['TITLE'])
    tfidf_column = vec_tfidf.get_feature_names_out()
    
    X_train_title = pd.DataFrame(vec_tfidf.transform(train['TITLE']).toarray(), train.index.values, columns=tfidf_column)
    X_valid_title = pd.DataFrame(vec_tfidf.transform(valid['TITLE']).toarray(), valid.index.values, columns=tfidf_column)
    X_test_title  = pd.DataFrame(vec_tfidf.transform(test['TITLE']).toarray(), test.index.values, columns=tfidf_column)

    # FIXME one-hotエンコーディングがそれぞれのデータで異なる可能性
    X_train_pub_and_host = pd.get_dummies(train[['PUBLISHER', 'HOSTNAME']])
    X_valid_pub_and_host = pd.get_dummies(valid[['PUBLISHER', 'HOSTNAME']])
    X_test_pub_and_host = pd.get_dummies(test[['PUBLISHER', 'HOSTNAME']])

    # FIXME one-hotエンコーディングがそれぞれのデータで異なる可能性
    Y_train = pd.get_dummies(train['CATEGORY']).to_numpy(dtype='int32')
    Y_train = np.argmax(Y_train, axis=1)
    Y_valid = pd.get_dummies(valid['CATEGORY']).to_numpy(dtype='int32')
    Y_valid = np.argmax(Y_valid, axis=1)
    Y_test = pd.get_dummies(test['CATEGORY']).to_numpy(dtype='int32')
    Y_test = np.argmax(Y_test, axis=1)

    X_train = pd.concat([X_train_title, X_train_pub_and_host], axis=1)
    X_valid = pd.concat([X_valid_title, X_valid_pub_and_host], axis=1)
    X_test = pd.concat([X_test_title, X_test_pub_and_host], axis=1)

    feature_columns = X_train.columns.values
    
    train_size = X_train.shape[0]
    feature_size = X_train.shape[1]
    class_size = 4 # # of category

    # ロジスティック回帰モデル
    class LogisticRegression(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LogisticRegression, self).__init__()
            self.l1 = torch.nn.Linear(input_dim, output_dim).cuda()

        def forward(self, x):
            x = self.l1(x)
            return x

    model = LogisticRegression(feature_size, class_size)
    model.load_state_dict(torch.load('result/knock52/model.pth'))

    weight = model.l1.weight.detach().cpu()
    category = ['b', 'e', 'm', 't']
    for c, row in zip(category, weight):
        print(f'category: {c}')
        top10_arg = torch.argsort(row)[-10:]
        worst10_arg = torch.argsort(row)[:10]
        print('top10: {}'.format(feature_columns[top10_arg]))
        print('worst10: {}'.format(feature_columns[worst10_arg]))



    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
