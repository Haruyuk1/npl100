from argparse import ArgumentParser
from io import TextIOWrapper
import os


# コマンドライン引数 -head: int


def main(f: TextIOWrapper):
    source_txt = "data/popular-names.txt"
    parser = ArgumentParser()
    parser.add_argument("-head")
    args = parser.parse_args()

    try:
        N = int(args.head)
    except:
        print("N must be integer")

    if N < 0:
        print("N must not be negative")
        return
    
    with open(source_txt, mode="r") as f_source:
        head_N_lines = f_source.readlines()[:N]
    for line in head_N_lines:
        print(line, end="")
    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)