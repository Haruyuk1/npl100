import json
import os
import re
from io import TextIOWrapper

import pandas as pd


def main(f: TextIOWrapper):
    source_json = "data/jawiki-country.json"

    df = pd.read_json(source_json, lines=True)
    title, text = df.query('title=="イギリス"').values[0]
    
    lines = text.split("\n")
    
    category_line = list(filter(lambda line: re.search("\[Category:", line), lines))
    print(category_line)

    return



if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
