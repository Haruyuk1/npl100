import os
import pickle
from cgi import test
from io import TextIOWrapper
from msilib import type_valid

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    
    batch_size = 8
    train_dataset = TensorDataset(X_train, Y_train)
    valid_dataset = TensorDataset(X_valid, Y_valid)
    test_dataset = TensorDataset(X_test, Y_test)
    train_dataloader = DataLoader(train_dataset, batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size)

    num_epoch = 50
    lr = 0.001
    hp_lambdas = [10**i for i in range(-4, 5)]
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_size = X_train.shape[0]
    valid_size = X_valid.shape[0]
    test_size = X_test.shape[0]
    train_acc = []
    valid_acc = []
    test_acc = []


    # lambda loop
    for hp_lambda in tqdm(hp_lambdas):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=hp_lambda)
        
        # 重みリセット
        torch.nn.init.normal_(model.l1.weight)

        # 学習
        model.train()
        for i in tqdm(range(1, num_epoch + 1), desc='epoch'):
            epoch_loss = 0
            num_correct = 0
            for x, y in train_dataloader:
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss

        # 推論
        model.eval()

        # テストデータ
        train_num_correct = 0
        for x, y in train_dataloader:
            output = model(x)
            predict = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
            train_num_correct += int(torch.sum(predict == y))
        train_acc.append(train_num_correct/train_size)

        # 検証データ
        valid_num_correct = 0
        for x, y in valid_dataloader:
            output = model(x)
            predict = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
            valid_num_correct += int(torch.sum(predict == y))
        valid_acc.append(valid_num_correct/valid_size)

        # テストデータ
        test_num_correct = 0
        for x, y in test_dataloader:
            output = model(x)
            predict = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
            test_num_correct += int(torch.sum(predict == y))
        test_acc.append(test_num_correct/test_size)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.plot(hp_lambdas, train_acc)
    ax2.plot(hp_lambdas, valid_acc)
    ax3.plot(hp_lambdas, test_acc)

    plt.show()

                




    # torch.save(model.state_dict(), 'result/knock52/model.pth')
    

    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
