from io import TextIOWrapper
import os


def main(f: TextIOWrapper):
    source_txt = "data/popular-names.txt"
    with open(source_txt, mode="r") as f_source:
        lines: list[str] = f_source.readlines()
        print(list(map(lambda line: line.replace("\t", " "), lines)))

    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)