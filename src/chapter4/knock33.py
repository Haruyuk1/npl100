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
    
    ans: set = set()
    for sentense in res:
        if len(sentense) < 3:
            continue
        for i in range(len(sentense)-2):
            if sentense[i+1]['surface'] == 'の' and sentense[i]['pos'] == sentense[i+2]['pos'] == '名詞':
                a = sentense[i]['surface']
                b = sentense[i+2]['surface']
                ans.add((a, b))

    print(len(ans))
    pprint.pprint(list(ans)[:10])




    return


if __name__ == "__main__":
    # hogehoge.pyだった場合、result/hogehoge.txtへのTextIOWrapperをmainに渡す
    target_file_name: str = os.path.basename(__file__).replace(".py", "")
    target_folder_name: str = "result"
    target_file_path: str = target_folder_name + "/" + target_file_name + ".txt"   
    with open(target_file_path, mode="w", encoding="utf-8") as f:
        main(f)