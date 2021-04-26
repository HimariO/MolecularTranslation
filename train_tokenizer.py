import os
import fire
import pandas as pd
from pathlib import Path
import tokenizers

from tokenizers.trainers import BpeTrainer
from tokenizers.implementations import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceUnigramTokenizer
from tokenizers import Tokenizer


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
    tokenizer.save("./checkpoints/bms_tokenizer.json")

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
    })