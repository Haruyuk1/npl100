from io import TextIOWrapper
import os
import pandas as pd


def main(f: TextIOWrapper):
    dataset_path = 'data/newsCorpora.csv'

    df = pd.read_table(dataset_path, header=None, names=["ID","TITLE","URL","PUBLISHER","CATEGORY","STORY","HOSTNAME","TIMESTAMP"], index_col='ID')
    # filter
    df = df.query('PUBLISHER == ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]')
    
    # shuffle
    df = df.sample(frac=1)

    # partition
    train_end = int(df.shape[0]*0.8)
    valid_end = int(df.shape[0]*0.9)
    train_df = df[:train_end]
    valid_df = df[train_end:valid_end]
    test_df = df[valid_end:]

    # export
    train_df.to_csv('data/train.txt', sep='\t', header=None)
    valid_df.to_csv('data/valid.txt', sep='\t', header=None)
    test_df.to_csv('data/test.txt', sep='\t', header=None)

    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)