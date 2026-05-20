import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import faiss
import argparse
import numpy as np
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def mine_negatives(entity_file, train_file, output_path, k=50, model_name="openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    print("Loading data")
    entity_data = sorted(load_json(entity_file), key=lambda x: x['id'])
    train_data = load_json(train_file)

    print(f"Encoding {len(entity_data)} Entities")
    ent_embeds = []
    batch_size = 512

    with torch.no_grad():
        for i in tqdm(range(0, len(entity_data), batch_size)):
            batch = entity_data[i:i + batch_size]

            texts = []
            for e in batch:
                text = f"{e['entity_name']} {e.get('instance', '')} {e.get('desc', '')}"
                texts.append(text.strip())

            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            embeds = model.get_text_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            ent_embeds.append(embeds.cpu().numpy())

    ent_embeds = np.concatenate(ent_embeds, axis=0)

    print("Building Index...")
    index = faiss.IndexFlatIP(ent_embeds.shape[1])
    index.add(ent_embeds)

    print("Mining Hard Negatives...")
    hard_neg_map = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(train_data), batch_size)):
            batch = train_data[i:i + batch_size]
            texts = [m['mentions'] + " " + m['sentence'] for m in batch]

            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            men_embeds = model.get_text_features(**inputs)
            men_embeds = men_embeds / men_embeds.norm(dim=-1, keepdim=True)

            D, I = index.search(men_embeds.cpu().numpy(), k)

            for j, indices in enumerate(I):
                sample = batch[j]
                sample_id = sample['id'] 
                ground_truth_qid = sample.get('answer')  

                neg_list = []
                for idx in indices:
                    retrieved_ent = entity_data[idx]

                    if retrieved_ent['qid'] != ground_truth_qid:
                        neg_list.append(retrieved_ent['id'])

                hard_neg_map[str(sample_id)] = neg_list[:10]

    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(hard_neg_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine hard negatives for EVEMEL/HVMCM.")
    parser.add_argument("--entity_file", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--model_name", default=os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"))
    args = parser.parse_args()

    mine_negatives(args.entity_file, args.train_file, args.output_path, args.k, args.model_name)
