from io import TextIOWrapper
import os


def main(f: TextIOWrapper):
    statement = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
    statement = statement.replace(".", "")
    words = statement.split(" ")
    extract_one_index = [1,5,6,7,8,9,15,16,19]
    ans: dict = dict()
    for i, word in enumerate(words, 1):
        if i in extract_one_index:
            ans[word[0]] = i
        else:
            ans[word[:2]] = i
    print(ans)
    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)