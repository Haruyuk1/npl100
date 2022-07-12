import json
import os
from io import TextIOWrapper
import pprint
import re

import pandas as pd

def main(f: TextIOWrapper):
    source_json = "data/jawiki-country.json"

    df = pd.read_json(source_json, lines=True)
    title, text = df.query('title=="イギリス"').values[0]
    
    match: re.Match = re.search(r"\{\{基礎情報.*?\n(.*?)\n\}\}", text, flags=re.DOTALL+re.MULTILINE)
    template = match.group(1)
    
    # ファイル除去
    match: str = re.sub(
        r'''
        \[\[
        (ファイル:|File:)
        (.+?)
        \|
        (.+?)
        \|
        (.+?)
        \]\]
        ''', r'\2' ,template, flags=re.DOTALL+re.MULTILINE+re.VERBOSE
    )

    # template:lang除去
    match: str = re.sub(
        r'''
        \{\{
        lang\|
        (.+?)\|
        (.+?)
        \}\}
        ''', r'\2', match, flags=re.DOTALL+re.MULTILINE+re.VERBOSE
    )

    # HTMLタグ除去
    match: str = re.sub(
        r'''
        <
        .+?
        >
        ''', '', match, flags=re.DOTALL+re.MULTILINE+re.VERBOSE
    )

    # 内部リンク除去
    match: str = re.sub(
        r'''
        \[\[
        ([^\|]*?)
        \|??
        ([^\|]+?)
        \]\]
        ''', r"\2", match, flags=re.DOTALL+re.VERBOSE+re.MULTILINE
    )

    # 強調表示除去
    match: str = re.sub(
        r'''
        (\'{2,5})
        (.+?)
        \1
        ''', r"\2", match, flags=re.MULTILINE+re.VERBOSE
    )

    match: list = re.findall(
        r'''
        \|
        (.*?)
        \s*
        =
        \s*
        (.*?)
        (?:
          (?=\n\|)  # 改行(\n)+'|'の手前(肯定の先読み)
        | (?=\n$)   # または、改行(\n)+終端の手前(肯定の先読み)
        )
        \n
        ''', match, flags=re.DOTALL+re.MULTILINE+re.VERBOSE)
    
    res: dict = dict()
    for key, value in match:
        res[key] = value

    nation_flag_file = res['国旗画像']
    URL = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=imageinfo&list=&meta=&titles=File%3A{nation_flag_file}&iiprop=url"
    
    import requests
    response = requests.get(URL).json()
    pprint.pprint(response)
    pages = response['query']['pages']
    for page in pages:
        url = pages[page]['imageinfo'][0]['url']
    print(url)

    

    return



if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)
