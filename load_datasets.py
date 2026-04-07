from datasets import load_dataset

sst2 = load_dataset("stanfordnlp/sst2")
snli = load_dataset("stanfordnlp/snli")

snli = snli.filter(lambda x: x["label"] != -1)

print(sst2)
print(snli)