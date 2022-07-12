from io import TextIOWrapper
import os
import pprint
import MeCab


def main(f: TextIOWrapper):
    source_mecab = "data/neko.txt.mecab"
    
    with open(source_mecab, mode="r", encoding="utf-8") as f_mecab:
        lines = f_mecab.readlines()

    res: list[list[dict]] = []
    keitaiso: list[dict] = []

    for line in lines:
        surface, *others = line.split("\t")
        if others:
            others = others[0].split(',')
            base = others[6]
            pos = others[0]
            pos1 = others[1]
            tmp_dict = dict()
            tmp_dict['surface'] = surface
            tmp_dict['base'] = base
            tmp_dict['pos'] = pos
            tmp_dict['pos1'] = pos1
            keitaiso.append(tmp_dict)
        if surface == "EOS\n" and keitaiso:
            res.append(keitaiso)
            keitaiso = []
    
    verbs_base = set()
    for sentense in res:
        for word in sentense:
            if word['pos'] == '動詞':
                verbs_base.add(word['base'])
    
    print(len(verbs_base))
    print(list(verbs_base)[:30])

    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)