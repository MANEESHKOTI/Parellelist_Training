
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class TinyStoriesDataset(Dataset):
    def __init__(self, split="train", max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)
        self.max_length = max_length
    def __len__(self): return 1000 # Capped for benchmark speed
    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        enc = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["input_ids"].squeeze(0)

def get_dataloader(batch_size=4, split="train", max_length=1024):
    ds = TinyStoriesDataset(split=split, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"), drop_last=True)
