from io import TextIOWrapper
import os
from knock05 import makeNGram

def main(f: TextIOWrapper):
    parapara = "paraparaparadise"
    para = "paragraph"
    X = set(makeNGram(parapara, 2))
    Y = set(makeNGram(para, 2))
    sumset = X | Y
    multset = X & Y
    subset = X - Y
    print(sumset)
    print(multset)
    print(subset)
    if "se" in X: print("X includes se")
    if "se" in Y: print("Y includes se")
    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)