from io import TextIOWrapper
import os


def main(f: TextIOWrapper):
    def cipher(s: str):
        def convert(c: str):
            return chr(219-ord(c)) if c.islower() else c
        return "".join(list(map(lambda c: convert(c), s)))

    message = "This Is A Test Message."
    print(cipher(message))
    print(cipher(cipher(message)))
    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)