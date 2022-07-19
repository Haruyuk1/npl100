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
    
    batch_size_list = [8, 16, 32, 64]
    lr_list = [10**i for i in range(-5, -1)]
    lambda_list = [10**i for i in range(-6, -2)]
    optimizers = [torch.optim.SGD, torch.optim.Adam, torch.optim.Adagrad, 
                  torch.optim.RMSprop, torch.optim.Adadelta]
    train_dataset = TensorDataset(X_train, Y_train)
    valid_dataset = TensorDataset(X_valid, Y_valid)

    num_epoch = 20
    criterion = torch.nn.CrossEntropyLoss()

    valid_size = X_valid.shape[0]
    valid_acc = []
    conditions = []

    for batch_size in batch_size_list:
        for lr in lr_list:
            for hp_lambda in lambda_list:
                for optimizerClass in optimizers:
                    # データローダー準備
                    train_dataloader = DataLoader(train_dataset, batch_size)
                    valid_dataloader = DataLoader(valid_dataset, batch_size)


                    # 条件を記憶
                    condition = \
                        f"batch:{batch_size}, lr:{lr}, lambda:{hp_lambda}, optimizer:{optimizerClass.__name__}"
                    conditions.append(condition)

                    # 重みリセット
                    torch.nn.init.normal_(model.l1.weight)

                    # 学習
                    optimizer = optimizerClass(model.parameters(), lr=lr, weight_decay=hp_lambda)
                    model.train()
                    for epoch in tqdm(range(1, num_epoch+1), leave=False, desc='epoch loop'):
                        for x, y in train_dataloader:
                            optimizer.zero_grad()
                            output = model(x)
                            loss = criterion(output, y)
                            loss.backward()
                            optimizer.step()
                            
                    # 検証
                    valid_num_correct = 0
                    model.eval()
                    for x, y in valid_dataloader:
                        output = model(x)
                        predict = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
                        valid_num_correct += int(torch.sum(predict == y))

                    print(condition)
                    print(f'valid_acc:{valid_num_correct/valid_size}')
                    valid_acc.append(valid_num_correct/valid_size)

    best_perfomance_index = int(torch.argmax(torch.FloatTensor(valid_acc)))
    print(f'best confition: {conditions[best_perfomance_index]}')
    print(f'valid_acc: {valid_acc[best_perfomance_index]}')         

    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
