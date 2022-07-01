from io import TextIOWrapper
import os

def makeNGram(seq, n: int):
    ans: list = []
    for s in range(len(seq)-n+1):
        ans.append(seq[s:s+n])
    return ans

def main(f: TextIOWrapper):
    statement = "I am an NLPer"
    char_bi_gram = makeNGram(statement, 2)
    word_bi_gram = makeNGram(statement.split(" "), 2)
    print(char_bi_gram)
    print(word_bi_gram)
    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)