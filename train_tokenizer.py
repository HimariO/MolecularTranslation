from pathlib import Path

from tokenizers.trainers import BpeTrainer
from tokenizers import ByteLevelBPETokenizer


def main():
    paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # Save files to disk
    tokenizer.save_model(".", "esperberto")


    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)