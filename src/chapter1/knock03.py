from io import TextIOWrapper
import os
import re


def main(f: TextIOWrapper):
    statement = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    word_lengths = list(map(lambda word: len(word), re.split("[¥ ,.]", statement)))
    word_lengths = list(filter(lambda l: l>0, word_lengths))
    print(word_lengths)
    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)