from io import TextIOWrapper
import os
import random


def main(f: TextIOWrapper):
    def randomSort(s: str):
        def innerSort(s: str):
            if len(s) <= 4:
                return s
            return s[0] + "".join(random.sample(s[1:-1], len(s[1:-1]))) + s[-1]
        words = s.split(" ")
        return " ".join(list(map(lambda word: innerSort(word), words)))

    message = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    print(randomSort(message))
    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)