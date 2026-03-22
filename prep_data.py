import spacy
from datasets import load_dataset
from tqdm import tqdm
import os

nlp = spacy.load("en_core_web_md")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

def anonymize(text):
    doc = nlp(text)
    ents = sorted([e for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT", "FAC", "WORK_OF_ART"}], 
                  key=lambda x: x.start_char, reverse=True)
    counter = 0
    for ent in ents:
        placeholder = f"<ENT_{ent.label_}_{counter}>"
        text = text[:ent.start_char] + placeholder + text[ent.end_char:]
        counter += 1
    return text

print("Anonymizing...")
anonymized = []
for ex in tqdm(dataset):
    full_text = ex["article"] + "\n\n" + ex["highlights"]
    anonymized.append(anonymize(full_text))

os.makedirs("data", exist_ok=True)
with open("data/anonymized_train.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(anonymized))

print("Done! Corpus size:", len(open("data/anonymized_train.txt").read()) // 1_000_000, "MB")
