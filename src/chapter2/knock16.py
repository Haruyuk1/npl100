from argparse import ArgumentParser
from io import TextIOWrapper
import os


# コマンドライン引数 -split N: int


def main(f: TextIOWrapper):
    source_txt = "data/popular-names.txt"
    parser = ArgumentParser()
    parser.add_argument("-split")
    args = parser.parse_args()

    try:
        N = int(args.split)
    except:
        print("N must be integer")
        return

    if N < 0:
        print("N must not be negative")
        return
    
    with open(source_txt, mode="r") as f_source:
        lines = f_source.readlines()
    
    # だいたいN等分割 https://qiita.com/keisuke-nakata/items/c18cda4ded06d3159109
    l = [(len(lines) + i) // N for i in range(N)]
    now_index = 0
    os.makedirs("result/knock16", exist_ok=True)
    for i, num in enumerate(l):
        with open(f"result/knock16/split{i}.txt", mode="w") as f_n:
            f_n.writelines(lines[now_index:now_index+num])
        now_index += num
    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)