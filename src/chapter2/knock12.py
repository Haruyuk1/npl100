from io import TextIOWrapper
import os



def main(f: TextIOWrapper):
    source_txt = "data/popular-names.txt"
    with open(source_txt, mode="r") as f_source:
        lines: list[str] = f_source.readlines()
    separated_lines = list(map(lambda line: line.split("\t"), lines))
    data_col_1 = [e[0]+"\n" for e in separated_lines]
    data_col_2 = [e[1]+"\n" for e in separated_lines]
    os.makedirs("result/knock12", exist_ok=True)
    with open("result/knock12/col1.txt", mode="w", encoding="utf-8") as f_col1:
        f_col1.writelines(data_col_1)
    with open("result/knock12/col2.txt", mode="w", encoding="utf-8") as f_col2:
        f_col2.writelines(data_col_2)


    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)