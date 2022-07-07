import os
from io import TextIOWrapper

import pandas as pd


def main(f: TextIOWrapper):
    source_txt = "data/popular-names.txt"
    df = pd.read_table(source_txt, header=None)
    print(df[0].value_counts())

    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
