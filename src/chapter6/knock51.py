import os
from io import TextIOWrapper
import pickle
import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def main(f: TextIOWrapper):
    train_path = 'data/train.txt'
    valid_path = 'data/valid.txt'
    test_path = 'data/test.txt'

    doc2vec_path = 'data/apnews_dbow/doc2vec.bin'

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
    
    X_train_title = vec_tfidf.transform(train['TITLE']).toarray()
    X_valid_title = vec_tfidf.transform(valid['TITLE']).toarray()
    X_test_title = vec_tfidf.transform(test['TITLE']).toarray()

    X_train_pub_and_host = pd.get_dummies(train[['PUBLISHER', 'HOSTNAME']]).to_numpy(dtype='float64')
    X_valid_pub_and_host = pd.get_dummies(valid[['PUBLISHER', 'HOSTNAME']]).to_numpy(dtype='float64')
    X_test_pub_and_host = pd.get_dummies(test[['PUBLISHER', 'HOSTNAME']]).to_numpy(dtype='float64')

    Y_train = pd.get_dummies(train['CATEGORY']).to_numpy(dtype='int32')
    Y_train = np.argmax(Y_train, axis=1)
    Y_valid = pd.get_dummies(valid['CATEGORY']).to_numpy(dtype='int32')
    Y_valid = np.argmax(Y_valid, axis=1)
    Y_test = pd.get_dummies(test['CATEGORY']).to_numpy(dtype='int32')
    Y_test = np.argmax(Y_test, axis=1)

    X_train = np.concatenate([X_train_title, X_train_pub_and_host], 1)
    X_valid = np.concatenate([X_valid_title, X_valid_pub_and_host], 1)
    X_test = np.concatenate([X_test_title, X_test_pub_and_host], 1)

    X = (X_train, X_valid, X_test)
    Y = (Y_train, Y_valid, Y_test)

    pickle.dump(X, open('data/x.pkl', 'wb'))
    pickle.dump(Y, open('data/y.pkl', 'wb'))



    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
