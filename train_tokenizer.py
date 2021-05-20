# %%
import os
import fire
import pandas as pd
from pathlib import Path
from typing import List
import tokenizers

from tokenizers.trainers import BpeTrainer
from tokenizers.implementations import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceUnigramTokenizer
from tokenizers import Tokenizer

# %%


def handcraft():
    from tokenizers import models, normalizers, pre_tokenizers, decoders, Regex

    vocab = {
        "[UNK]": 0,
        "[CLS]":1,
        '[SEP]': 2,
        '[PAD]': 3,
        '[MASK]': 4,
        '(': 190,
        ')': 191,
        '+': 192,
        ',': 193,
        '-': 194,
        'InChI=1S/': 195,
        '/q': 196,
        '.': 197,
        ';': 198,
        '*': 199,
        '/p': 200,
        '/b': 5,
        '/c': 6,
        '/h': 7,
        '/i': 8,
        '/m': 9,
        '/s': 10,
        '/t': 11,
        '0': 12,
        '1': 13,
        '10': 14,
        '100': 15,
        '101': 16,
        '102': 17,
        '103': 18,
        '104': 19,
        '105': 20,
        '106': 21,
        '107': 22,
        '108': 23,
        '109': 24,
        '11': 25,
        '110': 26,
        '111': 27,
        '112': 28,
        '113': 29,
        '114': 30,
        '115': 31,
        '116': 32,
        '117': 33,
        '118': 34,
        '119': 35,
        '12': 36,
        '120': 37,
        '121': 38,
        '122': 39,
        '123': 40,
        '124': 41,
        '125': 42,
        '126': 43,
        '127': 44,
        '128': 45,
        '129': 46,
        '13': 47,
        '130': 48,
        '131': 49,
        '132': 50,
        '133': 51,
        '134': 52,
        '135': 53,
        '136': 54,
        '137': 55,
        '138': 56,
        '139': 57,
        '14': 58,
        '140': 59,
        '141': 60,
        '142': 61,
        '143': 62,
        '144': 63,
        '145': 64,
        '146': 65,
        '147': 66,
        '148': 67,
        '149': 68,
        '15': 69,
        '150': 70,
        '151': 71,
        '152': 72,
        '153': 73,
        '154': 74,
        '155': 75,
        '156': 76,
        '157': 77,
        '158': 78,
        '159': 79,
        '16': 80,
        '161': 81,
        '163': 82,
        '165': 83,
        '167': 84,
        '17': 85,
        '18': 86,
        '19': 87,
        '2': 88,
        '20': 89,
        '21': 90,
        '22': 91,
        '23': 92,
        '24': 93,
        '25': 94,
        '26': 95,
        '27': 96,
        '28': 97,
        '29': 98,
        '3': 99,
        '30': 100,
        '31': 101,
        '32': 102,
        '33': 103,
        '34': 104,
        '35': 105,
        '36': 106,
        '37': 107,
        '38': 108,
        '39': 109,
        '4': 110,
        '40': 111,
        '41': 112,
        '42': 113,
        '43': 114,
        '44': 115,
        '45': 116,
        '46': 117,
        '47': 118,
        '48': 119,
        '49': 120,
        '5': 121,
        '50': 122,
        '51': 123,
        '52': 124,
        '53': 125,
        '54': 126,
        '55': 127,
        '56': 128,
        '57': 129,
        '58': 130,
        '59': 131,
        '6': 132,
        '60': 133,
        '61': 134,
        '62': 135,
        '63': 136,
        '64': 137,
        '65': 138,
        '66': 139,
        '67': 140,
        '68': 141,
        '69': 142,
        '7': 143,
        '70': 144,
        '71': 145,
        '72': 146,
        '73': 147,
        '74': 148,
        '75': 149,
        '76': 150,
        '77': 151,
        '78': 152,
        '79': 153,
        '8': 154,
        '80': 155,
        '81': 156,
        '82': 157,
        '83': 158,
        '84': 159,
        '85': 160,
        '86': 161,
        '87': 162,
        '88': 163,
        '89': 164,
        '9': 165,
        '90': 166,
        '91': 167,
        '92': 168,
        '93': 169,
        '94': 170,
        '95': 171,
        '96': 172,
        '97': 173,
        '98': 174,
        '99': 175,
        'B': 176,
        'Br': 177,
        'C': 178,
        'Cl': 179,
        'D': 180,
        'F': 181,
        'H': 182,
        'I': 183,
        'N': 184,
        'O': 185,
        'P': 186,
        'S': 187,
        'Si': 188,
        'T': 189,
    }
    units = [k.replace('[', '\[').replace(']', '\]') for k in vocab.keys() if len(k) > 1] + ['.']
    units = sorted(units, key=lambda x: len(x), reverse=True)
    print("|".join(units))
    units_rex = Regex("|".join(units))

    class CustomDecoder:
        def decode(self, tokens: List[str]) -> str:
            return "".join(tokens)

    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=units_rex, behavior="isolated")
    tokenizer.decoder = decoders.Decoder.custom(CustomDecoder())
    
    # encoding = tokenizer.encode("InChI=1S/C21H30O4/c1-12(22)25-14-6-8-20(2)13(10-14)11-17(23)19-15-4-5-18(24)21(15,3)9-7-16(19)20/h13-16,19H,4-11H2,1-3H3/t13-,14+,15+,16-,19-,20+,21+/m1/s1")
    # encoding = tokenizer.encode("InChI=1S/C11H7BrFN2/c1-6-7(4-14)5-15-11-9(13)3-2-8(12)10(6)11/h2-3,5,9,11H,1H2/q-1")
    # encoding = tokenizer.encode("InChI=1S/C20H15F3N10O3/c1-10(33-9-28-15-14(33)17(35)32(4-3-24)19(36)31(15)2)16(34)30-13-8-25-7-12(29-13)11-5-26-18(27-6-11)20(21,22)23/h5-10H,4H2,1-2H3,(H,29,30,34)/t10-/m0/s1")
    # inchi_str = "InChI=1S/C9H10O2S.C7H12N2O4P.H2I/c1-6-3-4-7(5-8(6)12)9(10)11-2;10-6-4-2-1-3-5(6)7(11)8-9-14(12)13;/h3-5,12H,1-2H3;3-4,7-12H,1-2H2;1H2/q;-1;+1"
    # inchi_str = "InChI=1S/C23H21FN2O3S.C7H9NO2S/c1-16-7-13-20(14-8-16)30(28,29)25-15-17(2)26(19-11-9-18(24)10-12-19)23(27)21-5-3-4-6-22(21)25;1-6-2-4-7(5-3-6)11(8,9)10/h3-14,17H,15H2,1-2H3;2-5H,1H3,(H2,8,9,10)"
    # inchi_str = "InChI=1S/C17H33FO5.ClH.FH.H2O/c1-14-11-16(3-4-17(14)19)13-23-10-8-21-6-5-20-7-9-22-12-15(2)18;;;/h14-17,19H,3-13H2,1-2H3;2*1H;1H2"
    # inchi_str = "InChI=1S/C13H19N3O.C11H14IN3O/c1-2-15-6-8-16(9-7-15)13(17)11-4-3-5-12(14)10-11;1-6(2)15-7(3)11(16)14-8-5-13-10(12)4-9(8)15/h3-5,10H,2,6-9,14H2,1H3;4-7,12H,1-3H3/p+1"
    inchi_str = "InChI=1S/C13H10ClF3N2O3S.ClH/c14-10-3-1-7(5-9(10)13(15,16)17)19-23(21,22)8-2-4-11(18)12(20)6-8;/h1-6,19-20H,18H2;1H/p-1"
    encoding = tokenizer.encode(inchi_str)
    print(encoding)
    print(encoding.ids)
    print(encoding.tokens)
    print(tokenizer.decode(encoding.ids))
    print(inchi_str)
    # tokenizer.save("./checkpoints/handcraft_tokenizer.json")


def dump_inchi_txt(anno_csv, output_dir, splits=5):
    anno_df = pd.read_csv(anno_csv)
    inchi_strs = []
    for _, row in anno_df.iterrows():
        inchi_strs.append(row.InChI + "\n")
    os.makedirs(output_dir, exist_ok=True)
    inchi_splits = [inchi_strs[i::splits] for i in range(splits)]
    for i, inchis in enumerate(inchi_splits):
        txt_path = os.path.join(output_dir, f"bms_inchi.{i}.txt")
        with open(txt_path, mode='w') as f:
            f.writelines(inchis)


def train(txt_dir):
    paths = [str(x) for x in Path(txt_dir).glob("**/*.txt")]
    print(paths)

    # Initialize a tokenizer
    # tokenizer = BertWordPieceTokenizer()
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(
        files=paths,
        vocab_size=52000,
        min_frequency=2,
        special_tokens=[
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]"
        ])

    # Save files to disk
    # tokenizer.save_model("./checkpoints", "bms_tokenizer")
    tokenizer.save("./checkpoints/_bms_tokenizer.json")

    # trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    # tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)

def test_run(tk_model_file):
    # tokenizer = ByteLevelBPETokenizer(merges, vocab)
    tokenizer = Tokenizer.from_file(tk_model_file)  # ./checkpoints/bms_tokenizer.json
    inchi = (
        "InChI=1S/C23H25FN8O2/c1-13(28-23(33)20-15(21(25)27-12-26-20)11-32-6-2-3-7-32)17-9-19(34-31-17)22-29-16-5-4-14(10-24)8-18(16)30-22"
        "/h4-5,8-9,12-13H,2-3,6-7,10-11H2,1H3,(H,28,33)(H,29,30)(H2,25,26,27)/t13-/m0/s1")
    tkid = tokenizer.encode(f"[CLS]{inchi}[SEP]")
    print(tkid)
    print(tkid.tokens)
    print(tkid.ids)
    import pdb; pdb.set_trace()

    tkid = tokenizer.encode_batch([f"[CLS]{inchi}[SEP]", f"[CLS]{inchi}[SEP]"])
    print(tkid)


if __name__ == '__main__':
    fire.Fire({
        'dump': dump_inchi_txt,
        'train': train,
        'test_run': test_run,
        'handcraft': handcraft,
    })
# %%
