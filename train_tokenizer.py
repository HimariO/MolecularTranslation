import os
import fire
import pandas as pd
from pathlib import Path

from tokenizers.trainers import BpeTrainer
from tokenizers import ByteLevelBPETokenizer


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
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(
        files=paths,
        vocab_size=52_000,
        min_frequency=2,
        special_tokens=[
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]"
        ])

    # Save files to disk
    tokenizer.save_model("./checkpoints", "bms_tokenizer")

    # trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    # tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)


if __name__ == '__main__':
    fire.Fire({
        'dump': dump_inchi_txt,
        'train': train,
    })