import os
import pickle
from io import TextIOWrapper

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def main(f: TextIOWrapper):
    X_train, X_valid, X_test = pickle.load(open('data/x.pkl', 'rb'))
    Y_train, Y_valid, Y_test = pickle.load(open('data/y.pkl', 'rb'))

    X_train, X_valid, X_test = torch.cuda.FloatTensor(X_train), torch.cuda.FloatTensor(X_valid), torch.cuda.FloatTensor(X_test)
    Y_train, Y_valid, Y_test = torch.cuda.LongTensor(Y_train), torch.cuda.LongTensor(Y_valid), torch.cuda.LongTensor(Y_test)

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

    batch_size = 8
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # 評価
    model.eval()

    test_size = X_test.shape[0]
    test_num_correct = 0
    test_predicts = []

    for x, y in test_dataloader:
        output = model(x)
        predict = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
        test_predicts.append(predict)
        test_num_correct += torch.sum(predict == y)
    
    test_predicts = torch.concat(test_predicts, dim=0)

    labels = ['b', 'e', 'm', 't']
    report = classification_report(Y_test.cpu(), test_predicts.cpu())

    print(report)
    


    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
