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
    
    freq: dict = dict()
    for sentense in res:
        for word in sentense:
            if word['pos'] == '記号':
                continue
            if word['surface'] in freq:
                freq[word['surface']] += 1
            else:
                freq[word['surface']] = 1
    
    freq_list = [(k, v) for k, v in freq.items()]
    freq_list = list(sorted(freq_list, key=lambda t: t[1], reverse=True))

    pprint.pprint(freq_list[:10])


    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)