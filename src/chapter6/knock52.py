from io import TextIOWrapper
from msilib import type_valid
import os
import pandas as pd
import pickle
import torch
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
    dataset = TensorDataset(X_train, Y_train)
    num_epoch = 50
    batch_size = 8
    lr = 0.001
    dataloader = DataLoader(dataset, batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(1, num_epoch + 1):
        epoch_loss = 0
        num_correct = 0
        for x, y in tqdm(dataloader, leave=False, desc="epoch"):
            # 学習
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

            
            # 正誤分類
            predict = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
            num_correct += torch.sum(predict == y)

        print(f'--epoch {i}/{num_epoch} finished--')
        print(f'epoch_loss:{epoch_loss}, accuracy:{num_correct/train_size}')

    torch.save(model.state_dict(), 'result/knock52/model.pth')
    

    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)