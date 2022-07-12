import json
import os
from io import TextIOWrapper
import re

import pandas as pd

def main(f: TextIOWrapper):
    source_json = "data/jawiki-country.json"

    df = pd.read_json(source_json, lines=True)
    title, text = df.query('title=="イギリス"').values[0]
    
    lines = text.split("\n")
    res = []
    for line in lines:
        match = re.search(r"^(=+)\s*(.+?)\s*\1$", line)
        if match:
            num_equal = len(match.groups()[0])
            res.append((match.groups()[1], num_equal))
    
    print(res)

    return



if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
