import os
import copy
import json
import random
import torch
import pytorch_lightning as pl

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from urllib.parse import unquote

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_json_file(filepath):
    data = []
    if isinstance(filepath, str):
        with open(filepath, 'r', encoding='utf-8') as f:
            d = json.load(f)
            data.extend(d)
    elif isinstance(filepath, list):
        for path in filepath:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                data.extend(d)
    return data

class ToolAdaptiveMentionDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(ToolAdaptiveMentionDataModule, self).__init__()
        self.args = args

        processor = CLIPProcessor.from_pretrained(self.args.pretrained_model)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.feature_extractor

        self.train_file = self.args.data.train_file
        self.dev_file = self.args.data.dev_file
        self.test_file = self.args.data.test_file

        self.entity_file = self.args.data.entity
        self.qid2id_file = self.args.data.qid2id

        self.mention_img_folder = self.args.data.mention_img_folder
        self.kb_img_folder = self.args.data.kb_img_folder
        self.grounded_img_folder = self.args.data.grounded_img_folder

        assert os.path.exists(self.train_file), f"train_file not found: {self.train_file}"
        assert os.path.exists(self.dev_file), f"dev_file not found: {self.dev_file}"
        assert os.path.exists(self.test_file), f"test_file not found: {self.test_file}"
        assert os.path.exists(self.entity_file), f"entity file not found: {self.entity_file}"
        assert os.path.exists(self.qid2id_file), f"qid2id file not found: {self.qid2id_file}"

        assert os.path.isdir(self.mention_img_folder), f"mention_img_folder not found: {self.mention_img_folder}"
        assert os.path.isdir(self.kb_img_folder), f"kb_img_folder not found: {self.kb_img_folder}"
        assert os.path.isdir(self.grounded_img_folder), f"grounded_img_folder not found: {self.grounded_img_folder}"

        with open(self.qid2id_file, 'r', encoding='utf-8') as f:
            self.qid2id = json.loads(f.readline())

        self.raw_kb_entity = sorted(_load_json_file(self.entity_file), key=lambda x: x['id'])
        self.kb_entity = self.setup_dataset_for_entity(self.raw_kb_entity)
        self.kb_id2entity = {
            raw_ent['id']: ent for raw_ent, ent in zip(self.raw_kb_entity, self.kb_entity)
        }

        self.hard_neg_map = {}
        if hasattr(self.args.data, 'hard_neg_file') and self.args.data.hard_neg_file \
                and os.path.exists(self.args.data.hard_neg_file):
            print(f"Loading Hard Negatives from {self.args.data.hard_neg_file}...")
            with open(self.args.data.hard_neg_file, 'r', encoding='utf-8') as f:
                self.hard_neg_map = json.load(f)
        else:
            print("Warning: No hard negative file found. Training will proceed without hard negatives.")

        train_data = _load_json_file(self.train_file)
        dev_data = _load_json_file(self.dev_file)
        test_data = _load_json_file(self.test_file)

        self.train_data = self.setup_dataset_for_mention(train_data)
        self.val_data = self.setup_dataset_for_mention(dev_data)
        self.test_data = self.setup_dataset_for_mention(test_data)


    def is_valid_tool_output(self, tool_output):
        if tool_output is None:
            return False
        if not isinstance(tool_output, str):
            return False

        text = tool_output.strip()
        if len(text) == 0:
            return False

        lowered = text.lower()

        invalid_keywords = [
            "error:",
            "error code:",
            "error: failed to generate summary.",
            "traceback",
            "contentfilter",
            "content filter",
            "unsafe",
            "sensitive content",
            "'error':",
            '"error":',
        ]

        for kw in invalid_keywords:
            if kw in lowered:
                return False

        return True

    def build_augmented_text(self, mention, sentence, tool_call=None, tool_output=None):
        mention = unquote(str(mention)).strip()
        sentence = unquote(str(sentence)).strip()

        base_text = f"{mention} [SEP] {sentence}"

        valid_tool_output = self.is_valid_tool_output(tool_output)

        allowed_tools = {
            "Default_Case",
            "Text_Supplement",
            "Symbolic_Solver",
            "OCR_Augment",
            "Visual_Grounder",
        }

        if tool_call in allowed_tools and valid_tool_output:
            return f"{base_text} [SEP] {tool_output.strip()}"

        return base_text

    def resolve_mention_image_info(self, sample_dict):
        img_path = sample_dict.get('imgPath', '')
        response = sample_dict.get('response', {}) or {}
        tool_call = response.get('tool_call', None)
        processed_image = sample_dict.get('processed_image', None)

        if tool_call == "Visual_Grounder" and processed_image:
            return [processed_image], "grounded"

        if img_path != '':
            return [img_path], "mention"

        return [], "mention"

    def setup_dataset_for_entity(self, data):
        input_data = []

        for sample_dict in tqdm(data, desc='PreProcessing Entities'):
            sample_type = sample_dict['type']

            if sample_type != 'entity':
                continue

            entity = unquote(sample_dict.get('entity_name', '')).strip()
            attr = str(sample_dict.get('attr', '')).strip()
            instance = str(sample_dict.get('instance', '')).strip()
            desc = str(sample_dict.get('desc', '')).strip()

            aux_parts = [p for p in [attr] if p]
            aux_text = " ".join(aux_parts).strip()


            input_text = f"{entity} [SEP] {desc}"

            input_dict = self.tokenizer(
                input_text,
                padding='max_length',
                max_length=self.args.data.text_max_length,
                truncation=True
            )

            input_dict['img_list'] = sample_dict.get('image_list', [])
            input_dict['sample_type'] = 0

            if 'answer' in sample_dict:
                ans = sample_dict['answer']
                input_dict['answer'] = self.qid2id[ans] if ans in self.qid2id else 0

            input_data.append(input_dict)

        return input_data


    def setup_dataset_for_mention(self, data):
        input_data = []

        for sample_dict in tqdm(data, desc='PreProcessing Mentions'):
            sample_type = 1

            mention = sample_dict.get('mentions', '')
            sentence = sample_dict.get('sentence', '')

            response = sample_dict.get('response', {}) or {}
            tool_call = response.get('tool_call', None)
            tool_output = sample_dict.get('tool_output', None)

            input_text = self.build_augmented_text(
                mention=mention,
                sentence=sentence,
                tool_call=tool_call,
                tool_output=tool_output
            )

            input_dict = self.tokenizer(
                input_text,
                padding='max_length',
                max_length=self.args.data.text_max_length,
                truncation=True
            )

            img_list, image_source = self.resolve_mention_image_info(sample_dict)
            input_dict['img_list'] = img_list
            input_dict['image_source'] = image_source
            input_dict['sample_type'] = sample_type


            input_dict['tool_call'] = tool_call if tool_call is not None else "None"
            input_dict['has_valid_tool_output'] = self.is_valid_tool_output(tool_output)

            if 'id' in sample_dict:
                input_dict['qid'] = sample_dict['id']

            if 'answer' in sample_dict:
                if sample_dict['answer'] in self.qid2id:
                    input_dict['answer'] = self.qid2id[sample_dict['answer']]
                else:
                    continue

            if sample_dict.get('answer') == 'nil':
                continue

            input_data.append(input_dict)

        return input_data


    def choose_image(self, sample_type, img_list, is_eval=False, image_source="mention"):
        if len(img_list):
            img_name = random.choice(img_list)
            if is_eval:
                img_name = img_list[0]

            img_name = str(img_name).strip()

            try:
                if sample_type == 0:
                    img_root = self.args.data.kb_img_folder
                else:
                    if image_source == "grounded":
                        img_root = self.args.data.grounded_img_folder
                    else:
                        img_root = self.args.data.mention_img_folder


                stem = os.path.splitext(img_name)[0]
                img_name = stem + ".jpg"

                img_path = os.path.join(img_root, img_name)

                img = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)

            except Exception as e:
                print(f"[Image Load Warning] Failed to load image: {img_name} | source={image_source} | error={e}")
                pixel_values = torch.zeros((3, 224, 224))
        else:
            pixel_values = torch.zeros((3, 224, 224))

        return pixel_values

    def train_collator(self, samples):
        img_list, image_source_list, sample_type, input_dict_list = [], [], [], []
        pixel_values, gt_ent_id = [], []

        neg_img_list, neg_type, neg_input_dict_list, neg_pixel_values = [], [], [], []

        num_neg = getattr(self.args.data, 'num_hard_negatives', 1)

        for sample in samples:
            sample = copy.deepcopy(sample)

            img_list.append(sample.pop('img_list'))
            image_source_list.append(sample.pop('image_source', 'mention'))
            sample_type.append(sample.pop('sample_type'))
            gt_ent_id.append(sample.pop('answer'))

            qid = sample.pop('qid', None)

  
            sample.pop('tool_call', None)
            sample.pop('has_valid_tool_output', None)

            input_dict_list.append(sample)

            neg_candidates = []
            if qid is not None and str(qid) in self.hard_neg_map:
                candidate_int_ids = self.hard_neg_map[str(qid)]
                for cid in candidate_int_ids:
                    if cid in self.kb_id2entity:
                        neg_candidates.append(self.kb_id2entity[cid])

            chosen_negs = []
            if len(neg_candidates) > 0:
                if len(neg_candidates) >= num_neg:
                    chosen_negs = random.sample(neg_candidates, num_neg)
                else:
                    chosen_negs = (neg_candidates * (num_neg // len(neg_candidates) + 1))[:num_neg]
            else:
                while len(chosen_negs) < num_neg:
                    rand_ent = random.choice(self.kb_entity)
                    chosen_negs.append(rand_ent)

            for neg_ent in chosen_negs:
                neg_ent_cp = copy.deepcopy(neg_ent)
                neg_img_list.append(neg_ent_cp.pop('img_list'))
                neg_type.append(neg_ent_cp.pop('sample_type'))
                neg_input_dict_list.append(neg_ent_cp)

        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(
                self.choose_image(
                    sample_type=sample_type[idx],
                    img_list=img_list[idx],
                    image_source=image_source_list[idx],
                    is_eval=False
                )
            )

        input_dict = self.tokenizer.pad(
            input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        input_dict['pixel_values'] = torch.stack(pixel_values)

        ent_info_list = [copy.deepcopy(self.kb_id2entity[idx]) for idx in gt_ent_id]
        ent_img_list, ent_type, ent_input_dict_list, ent_pixel_values = [], [], [], []

        for ent_dict in ent_info_list:
            ent_img_list.append(ent_dict.pop('img_list'))
            ent_type.append(ent_dict.pop('sample_type'))
            ent_input_dict_list.append(ent_dict)

        for idx, _ in enumerate(ent_input_dict_list):
            ent_pixel_values.append(
                self.choose_image(
                    sample_type=ent_type[idx],
                    img_list=ent_img_list[idx],
                    image_source='kb',
                    is_eval=False
                )
            )

        ent_empty_img_flag = torch.tensor(
            [True if not len(_) else False for _ in ent_img_list],
            dtype=torch.bool
        )

        ent_input_dict = self.tokenizer.pad(
            ent_input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        ent_input_dict['pixel_values'] = torch.stack(ent_pixel_values)
        ent_input_dict['empty_img_flag'] = ent_empty_img_flag

        for k, v in ent_input_dict.items():
            input_dict[f'ent_{k}'] = v

        if len(neg_input_dict_list) > 0:
            for idx, _ in enumerate(neg_input_dict_list):
                neg_pixel_values.append(
                    self.choose_image(
                        sample_type=neg_type[idx],
                        img_list=neg_img_list[idx],
                        image_source='kb',
                        is_eval=False
                    )
                )

            neg_input_dict = self.tokenizer.pad(
                neg_input_dict_list,
                padding='max_length',
                max_length=self.args.data.text_max_length,
                return_tensors='pt'
            )
            neg_input_dict['pixel_values'] = torch.stack(neg_pixel_values)

            for k, v in neg_input_dict.items():
                input_dict[f'neg_{k}'] = v

        return input_dict

    def eval_collator(self, samples):
        img_list, image_source_list, sample_type, input_dict_list = [], [], [], []
        pixel_values, gt_ent_id = [], []

        for sample in samples:
            sample = copy.deepcopy(sample)

            img_list.append(sample.pop('img_list'))
            image_source_list.append(sample.pop('image_source', 'mention'))
            sample_type.append(sample.pop('sample_type'))
            gt_ent_id.append(sample.pop('answer'))

            sample.pop('qid', None)
            sample.pop('tool_call', None)
            sample.pop('has_valid_tool_output', None)

            input_dict_list.append(sample)

        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(
                self.choose_image(
                    sample_type=sample_type[idx],
                    img_list=img_list[idx],
                    image_source=image_source_list[idx],
                    is_eval=True
                )
            )

        input_dict = self.tokenizer.pad(
            input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        input_dict['pixel_values'] = torch.stack(pixel_values)
        input_dict['answer'] = torch.tensor(gt_ent_id, dtype=torch.long)

        return input_dict

    def entity_collator(self, samples):
        pixel_values, img_list, sample_type, input_dict_list = [], [], [], []

        for sample in samples:
            sample = copy.deepcopy(sample)
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            input_dict_list.append(sample)

        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(
                self.choose_image(
                    sample_type=sample_type[idx],
                    img_list=img_list[idx],
                    image_source='kb',
                    is_eval=True
                )
            )

        input_dict = self.tokenizer.pad(
            input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        input_dict['pixel_values'] = torch.stack(pixel_values)
        return input_dict


    def entity_dataloader(self):
        return DataLoader(
            self.kb_entity,
            batch_size=self.args.data.embed_update_batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=False,
            collate_fn=self.entity_collator
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.args.data.batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=True,
            collate_fn=self.train_collator
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.args.data.eval_batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=False,
            collate_fn=self.eval_collator
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.args.data.eval_batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=False,
            collate_fn=self.eval_collator
        )


