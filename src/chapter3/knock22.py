import json
import math
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
    categories = []
    for line in category_line:
        match: re.Match = re.search("\[\[Category:(.*?)(\|.*)*?\]\]", line)
        category = match.groups()[0]
        categories.append(category)
    print(categories)
    return



if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
