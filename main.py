!pip install transformers datasets torch cryptography

import pandas as pd

uploaded = pd.read_csv ("my_filename.csv")

filename = list(uploaded.keys())[0]

from datasets import Dataset
with open(filename, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]
dataset = Dataset.from_dict({"text": lines})

from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os

MODEL_PATH = "./bert-finance"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

args = TrainingArguments(
    output_dir=MODEL_PATH,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=1,
    prediction_loss_only=True,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator
)

print("Training started...")
trainer.train()
trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
print("Training complete. Model saved to", MODEL_PATH)

from transformers import pipeline
import random

class HoneyGenerator:
    def __init__(self, model_path: str):
        self.generator = pipeline("fill-mask", model=model_path, tokenizer=model_path)

    def generate(self, template: str, top_k: int = 5, n_variants: int = 5):
        if template.count("[MASK]") == 1:
            outputs = self.generator(template, top_k=top_k)
            return [o['sequence'] for o in outputs][:n_variants]
        else:
            variants = []
            for _ in range(n_variants):
                current = template
                while "[MASK]" in current:
                    outs = self.generator(current, top_k=top_k)
                    choice = random.choice(outs)
                    current = choice['sequence']
                variants.append(current)
            return variants

import hmac, hashlib

class DTE:
    def __init__(self, honey_messages):
        self.honey_messages = honey_messages
        self.n = len(honey_messages)

    def _index_from_key(self, key: str) -> int:
        hm = hmac.new(key.encode('utf-8'), b'dte-index', hashlib.sha256).digest()
        idx = int.from_bytes(hm, 'big') % self.n
        return idx

    def encode(self, true_key: str) -> int:
        return self._index_from_key(true_key)

    def decode(self, key: str) -> str:
        idx = self._index_from_key(key)
        return self.honey_messages[idx]

import base64
import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return kdf.derive(passphrase.encode('utf-8'))

def encrypt_index(index: int, passphrase: str = "default"):
    salt = os.urandom(16)
    key = derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    data = str(index).encode('utf-8')
    ct = aesgcm.encrypt(nonce, data, None)
    token = base64.urlsafe_b64encode(salt + nonce + ct).decode('utf-8')
    return token

def decrypt_index(token: str, passphrase: str = "default") -> int:
    raw = base64.urlsafe_b64decode(token)
    salt, nonce, ct = raw[:16], raw[16:28], raw[28:]
    key = derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    data = aesgcm.decrypt(nonce, ct, None)
    return int(data.decode('utf-8'))

print("\n Generating honey messages...")
generator = HoneyGenerator(MODEL_PATH)
template = "Debit of [MASK] made at [MASK] on [MASK]."
honey_messages = generator.generate(template, top_k=5, n_variants=5)

print("\n Generated honey messages:")
for m in honey_messages:
    print(" -", m)

dte = DTE(honey_messages)
user_key = input("\nEnter encryption key: ")
index = dte.encode(user_key)
token = encrypt_index(index, passphrase=user_key)
print(f"\n Encrypted index token: {token}")

attacker_key = input("Enter decryption key (try wrong keys too): ")
try:
    decrypted_index = decrypt_index(token, passphrase=attacker_key)
    output_message = honey_messages[decrypted_index]
except Exception as e:
    output_message = "(Decryption failed — random honey message)"

print(f"\n Decrypted/Honey message: {output_message}")
