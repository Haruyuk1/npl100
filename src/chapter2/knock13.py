from io import TextIOWrapper
import os


def main(f: TextIOWrapper):
    source_txt = "data/popular-names.txt"
    col1_txt = "result/col1.txt"
    col2_txt = "result/col2.txt"
    with open(col1_txt, mode="r", encoding="utf-8") as f_col1:
        col1 = f_col1.readlines()
    with open(col2_txt, mode="r", encoding="utf-8") as f_col2:
        col2 = f_col2.readlines()
    res = ["{}\t{}\n".format(name.replace("\n", ""), sex.replace("\n", "")) for name, sex in zip(col1, col2)]
    f.writelines(res)
    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)