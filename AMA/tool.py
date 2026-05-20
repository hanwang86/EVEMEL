import os

if os.getenv("EVEMEL_OFFLINE", "0") == "1":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

import sys
import json
import base64
import re
import torch
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR
from transformers import AutoProcessor, GroundingDinoForObjectDetection

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_glm import get_glm_response

from prompt.Prompt_Default_Case import prompt as PROMPT_DEFAULT
from prompt.Prompt_Text_Supplement import prompt as PROMPT_TEXT_SUPPLEMENT
from prompt.Prompt_Symbolic_Solver import prompt as PROMPT_SYMBOLIC
from prompt.Prompt_Visual_Grounder import prompt as PROMPT_VG
from prompt.Prompt_OCR_Augment import prompt as PROMPT_OCR
from prompt import prompt_dino_visual as prompt_visual_template

class ToolEngine:
    def __init__(self, args):
        self.args = args

        if hasattr(self.args, 'output_root') and self.args.output_root and not os.path.exists(self.args.output_root):
            os.makedirs(self.args.output_root)

        self.grounded_dir = os.path.join(self.args.output_root, 'grounded_images') if hasattr(self.args,
                                                                                              'output_root') and self.args.output_root else './grounded_images'
        if not os.path.exists(self.grounded_dir):
            os.makedirs(self.grounded_dir)

        self.dino_processor = None
        self.dino_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dino_model_path = getattr(
            args,
            'dino_model_path',
            os.getenv("GROUNDING_DINO_MODEL_PATH", "IDEA-Research/grounding-dino-base")
        )

        self.ocr_model = None

    def _load_dino_model(self):
        if self.dino_model is None:
            print(f"[ToolEngine] Loading GroundingDINO from {self.dino_model_path} on {self.device}...")
            try:
                self.dino_processor = AutoProcessor.from_pretrained(self.dino_model_path)
                self.dino_model = GroundingDinoForObjectDetection.from_pretrained(self.dino_model_path).to(self.device)
                print("[ToolEngine] GroundingDINO loaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to load GroundingDINO: {e}")

    def _load_ocr_model(self):
        if self.ocr_model is None:
            print("[ToolEngine] Loading PaddleOCR...")
            try:

                self.ocr_model = PaddleOCR()
            except Exception as e:
                print(f"[Error] Failed to load PaddleOCR: {e}")

    def _call_vlm_for_summary(self, prompt_text, image_path=None):
        content_list = []
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    img_base = base64.b64encode(img_file.read()).decode("utf-8")
                content_list.append({"type": "image_url", "image_url": {"url": img_base}})
            except Exception as e:
                print(f"[Error] Failed to read image for VLM: {e}")

        content_list.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content_list}]

        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = get_glm_response(
                    messages,
                    api_key=self.args.api_key,
                    model=self.args.policy_engine,
                    temperature=0.1,
                    max_tokens=512
                )

                if "Summary:" in response:
                    summary = response.split("Summary:")[-1].strip()
                elif "summary:" in response.lower():
                    summary = re.split("summary:", response, flags=re.IGNORECASE)[-1].strip()
                else:
                    summary = response.strip()

                summary = summary.strip('"').strip("'").replace("\n", " ")
                if summary:
                    return summary
            except Exception as e:
                print(f"[ToolEngine VLM Retry] API Error: {e}")
        return "Error: Failed to generate summary."

    def _generate_visual_prompt(self, mention, sentence):
        if not sentence: return mention
        try:
            prompt = prompt_visual_template.VISUAL_DESC_PROMPT.format(mention=mention, sentence=sentence)

            messages = [{"role": "user", "content": prompt}]
            response = get_glm_response(
                messages,
                api_key=self.args.api_key,
                model=self.args.text_policy_engine,
                temperature=0.1,
                max_tokens=30
            )
            clean_text = response.strip()
            if clean_text.lower().startswith("summary:"):
                clean_text = clean_text.split(":", 1)[1].strip()
            visual_prompt = clean_text.strip().strip(".")
            return visual_prompt if visual_prompt else mention
        except Exception as e:
            return mention

    def run_default_case(self, item):
        mention = item.get('mention', '')
        sentence = item.get('sentence', '')
        img_path = item.get('real_img_path')

        prompt = PROMPT_DEFAULT.format(mention_name=mention, mention_context=sentence)
        summary = self._call_vlm_for_summary(prompt, img_path)
        return {"tool_output": summary}

    def run_text_supplement(self, item):
        mention = item.get('mention', '')
        sentence = item.get('sentence', '')

        prompt = PROMPT_TEXT_SUPPLEMENT.format(mention_name=mention, mention_context=sentence)

        summary = self._call_vlm_for_summary(prompt, image_path=None)
        return {"tool_output": summary}

    def run_symbolic_solver(self, item):
        mention = item.get('mention', '')
        sentence = item.get('sentence', '')
        img_path = item.get('real_img_path')

        prompt = PROMPT_SYMBOLIC.format(mention_name=mention, mention_context=sentence)
        summary = self._call_vlm_for_summary(prompt, img_path)
        return {"tool_output": summary}

    def run_ocr_augment(self, item):
        mention = item.get('mention', '')
        sentence = item.get('sentence', '')
        img_path = item.get('real_img_path')

        ocr_text = ""
        if img_path and os.path.exists(img_path):
            if self.ocr_model is None:
                self._load_ocr_model()
            try:
                result = self.ocr_model.ocr(img_path)
                ocr_result = result[0]
                txts = []
                if ocr_result:
                    if isinstance(ocr_result, dict):
                        txts = ocr_result.get('rec_texts', [])
                    else:
                        txts = [line[1][0] for line in ocr_result]
                txts = [t for t in txts if t.strip()]
                ocr_text = " ".join(txts)
            except Exception as e:
                print(f"[OCR Tool Error] {e}")

        prompt = PROMPT_OCR.format(mention_name=mention, mention_context=sentence, ocr_extracted_text=ocr_text)
        summary = self._call_vlm_for_summary(prompt, img_path)
        return {"tool_output": summary}

    def run_visual_grounder(self, item):
        mention = item.get('mention', '')
        sentence = item.get('sentence', '')
        img_path = item.get('real_img_path')
        img_name = item.get('img_name', 'unknown.jpg')
        current_id = item.get('id', 'unknown')

        prompt = PROMPT_VG.format(mention_name=mention, mention_context=sentence)
        summary = self._call_vlm_for_summary(prompt, img_path)

        res = {"tool_output": summary}

        if img_path and os.path.exists(img_path):
            if self.dino_model is None:
                self._load_dino_model()
            try:
                image = Image.open(img_path).convert("RGB")

                text_prompt = f"{mention}."

                inputs = self.dino_processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.dino_model(**inputs)

                target_sizes = torch.tensor([image.size[::-1]])
                results = self.dino_processor.image_processor.post_process_object_detection(
                    outputs, threshold=0.25, target_sizes=target_sizes
                )[0]

                if results["scores"].numel() > 0:
                    max_idx = results["scores"].argmax()
                    box = results["boxes"][max_idx].tolist()
                    box_int = [int(coord) for coord in box]

                    draw = ImageDraw.Draw(image)
                    draw.rectangle(box_int, outline="red", width=4)

                    safe_mention = "".join([c if c.isalnum() else "_" for c in mention])[:20]
                    grounded_filename = f"{current_id}_{os.path.splitext(img_name)[0]}_grounded_{safe_mention}.jpg"
                    save_path = os.path.join(self.grounded_dir, grounded_filename)

                    image.save(save_path)

                    res["grounded_image_path"] = save_path
            except Exception as e:
                print(f"[Visual Grounder Error] Failed to draw box: {e}")

        return res
