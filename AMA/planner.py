import base64
import os
import sys
import json
import argparse
import re
import ast
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_glm import get_glm_response
from prompt import Prompt_Planner as prompt_template
from tool import ToolEngine

def parse_args():
    parser = argparse.ArgumentParser(description="WikiDiverse Entity Linking Planner")

    parser.add_argument('--dataset', type=str, default='WikiDiverse',
                        choices=['WikiDiverse', 'WikiMEL', 'RichpediaMEL'],
                        help='Dataset name')

    parser.add_argument('--data_root', type=str, default=None, help='Root directory for data')
    parser.add_argument('--result_data_root', type=str, default='./result', help='Directory for results')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Run mode')

    parser.add_argument('--image_dir', type=str, default=None, help='Image directory name')
    parser.add_argument('--KB_image_dir', type=str, default='kb_entity.json', help='KB directory name')
    parser.add_argument('--output_root', type=str, default= None, help='Output directory')
    parser.add_argument('--api_key', type=str, default=os.getenv('ZHIPUAI_API_KEY', ''),
                        help='ZhipuAI API Key. Defaults to the ZHIPUAI_API_KEY environment variable.')
    parser.add_argument('--policy_engine', type=str, default="glm-4.6v-flash", help='engine for module prediction')
    parser.add_argument('--text_policy_engine', type=str, default="glm-4-flash-250414", help='engine for text supplement')
    parser.add_argument('--policy_temperature', type=float, default=0., help='temperature')
    parser.add_argument('--policy_max_tokens', type=int, default=3072, help='max tokens')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to process')
    parser.add_argument('--stage', type=str, default='all', choices=['planner', 'executor', 'all'],
                        help='AMA stage to run')
    return parser.parse_args()

def get_mode_paths(args):
    dataset_result_dir = os.path.join(args.result_data_root, args.dataset)

    planner_dir = os.path.join(dataset_result_dir, "Planner")
    execution_dir = os.path.join(dataset_result_dir, "Execution")

    if not os.path.exists(planner_dir):
        os.makedirs(planner_dir)
    if not os.path.exists(execution_dir):
        os.makedirs(execution_dir)

    prefix = args.file_prefix
    mode = args.mode

    return {

        'source_data': os.path.join(args.data_root, f"{prefix}_{mode}.json"),

        'planner_out': os.path.join(planner_dir, f"{prefix}_{mode}_planner_results.json"),
        'planner_running': os.path.join(planner_dir, f"{prefix}_{mode}_planner_running.jsonl"),
        'planner_input_running': os.path.join(planner_dir, f"{prefix}_{mode}_planner_inputs_running.jsonl"),

        'final_out': os.path.join(execution_dir, f"{prefix}_{mode}_with_tools.json"),
        'executor_running': os.path.join(execution_dir, f"{prefix}_{mode}_tools_running.jsonl"),

        'output_dir': dataset_result_dir
    }

def init_path_config(args):

    DATASET_CONFIG = {
        'WikiDiverse': {
            'root': '../../data/WikiDiverse',
            'prefix': 'WikiDiverse',
            'image_dir': 'mention_image/mention_image'
        },
        'WikiMEL': {
            'root': '../../data/WikiMEL',
            'prefix': 'WIKIMEL',
            'image_dir': 'mention_image/mention_image'
        },
        'RichpediaMEL': {
            'root': '../../data/RichpediaMEL',
            'prefix': 'RichpediaMEL',
            'image_dir': 'mention_image'
        }
    }

    config = DATASET_CONFIG[args.dataset]

    if args.data_root is None:
        args.data_root = config['root']

    if args.image_dir is None:
        args.image_dir = config['image_dir']

    args.file_prefix = config['prefix']

    specific_result_dir = os.path.join(args.result_data_root, args.dataset)

    args.output_root = os.path.join(specific_result_dir, "Execution")

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    return args

def force_jpg_path(root_dir, img_name):
    if not img_name:
        return "", False, ""

    base_name = os.path.splitext(img_name)[0]

    forced_name = base_name + ".jpg"

    full_path = os.path.join(root_dir, forced_name)

    exists = os.path.isfile(full_path)

    return full_path, exists, forced_name

def _norm_key_value(value):
    if value is None:
        return ""
    return str(value).strip()

def record_key(item):
    return (
        _norm_key_value(item.get('id')),
        _norm_key_value(item.get('mentions') or item.get('mention')),
        _norm_key_value(item.get('answer')),
        _norm_key_value(item.get('sentence')),
    )

def _append_unique(items, value):
    if value and value not in items:
        items.append(value)

def force_jpg_path(root_dir, img_name):
    if not img_name:
        return "", False, ""

    clean_name = str(img_name).strip().replace("\\", "/")
    base_name, ext = os.path.splitext(clean_name)

    candidate_names = []
    _append_unique(candidate_names, clean_name)
    for suffix in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        _append_unique(candidate_names, base_name + suffix)
    if ext:
        _append_unique(candidate_names, clean_name)

    root_candidates = []
    _append_unique(root_candidates, root_dir)
    _append_unique(root_candidates, os.path.join(root_dir, "mention_image"))
    parent_dir = os.path.dirname(root_dir)
    _append_unique(root_candidates, parent_dir)
    _append_unique(root_candidates, os.path.join(parent_dir, "mention_image"))

    for candidate_name in candidate_names:
        if os.path.isabs(candidate_name) and os.path.isfile(candidate_name):
            return candidate_name, True, os.path.basename(candidate_name)

        for candidate_root in root_candidates:
            full_path = os.path.join(candidate_root, candidate_name)
            if os.path.isfile(full_path):
                try:
                    rel_name = os.path.relpath(full_path, root_dir)
                except ValueError:
                    rel_name = os.path.basename(full_path)
                return full_path, True, rel_name.replace("\\", "/")

    forced_name = base_name + ".jpg"
    return os.path.join(root_dir, forced_name), False, forced_name

class Planner:
    def __init__(self, args):
        self.args = args
        self.paths = get_mode_paths(args)
        self.image_root = os.path.join(args.data_root, args.image_dir)
        self.output_dir = self.paths['output_dir']
        self.tool_engine = ToolEngine(args)

    def load_data(self):
        print(f"Loading data from {self.paths['source_data']}...")
        with open(self.paths['source_data'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        if self.args.num_samples > 0:
            return data[:self.args.num_samples]
        else:
            return data

    def get_question_text(self, item):
        sentence = item.get('sentence', '')
        mentions = item.get('mention') or item.get('mentions') or ''
        raw_img_name = item.get('imgPath', '')

        image_path, has_image, img_name = force_jpg_path(self.image_root, raw_img_name)

        question_data = {
            "mentions": mentions,
            "sentence": sentence,
            "has_image": has_image
        }
        question_text = json.dumps(question_data, indent=2)
        return question_text, image_path, img_name, has_image

    def build_prompt(self, item):
        question_text, image_path, img_name, has_image = self.get_question_text(item)
        demo_prompt = prompt_template.prompt.strip()
        full_prompt = demo_prompt + "\n\n" + question_text
        return full_prompt, image_path, img_name, has_image

    def parse_llm_output(self, text):
        if not text: return None
        clean_text = text.strip()
        code_block_pattern = r"```(?:json|JSON)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, clean_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = clean_text.find('{')
            end = clean_text.rfind('}')
            if start != -1 and end != -1:
                json_str = clean_text[start:end + 1]
            else:
                json_str = clean_text
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                fixed_str = json_str.replace("None", "null").replace("True", "true").replace("False", "false")
                return json.loads(fixed_str)
            except:
                print(f"[Parse Error] JSON parsing failed. Content: {json_str[:50]}...")
                return text

    def convert_jsonl_to_json(self, source_file, target_file, desc):
        import re
        import ast

        print(f"Converting {desc} (.jsonl) to final JSON (with Robust Parsing)...")
        if not os.path.exists(source_file):
            print(f"[Warning] {source_file} not found, skipping conversion.")
            return

        unique_records_map = {}
        fixed_count = 0

        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:

                    record = json.loads(line)

                    raw_response = record.get('response')

                    if isinstance(raw_response, str):
                        clean_str = raw_response.strip()
                        parsed_obj = None

                        match = re.search(r"```(?:json|JSON)?\s*(.*?)\s*```", clean_str, re.DOTALL)
                        if match:
                            clean_str = match.group(1)

                        try:
                            parsed_obj = json.loads(clean_str)
                        except json.JSONDecodeError:

                            try:

                                fixed_json_str = clean_str.replace('\\n', '\n').replace('\\"', '"')
                                parsed_obj = json.loads(fixed_json_str)
                            except json.JSONDecodeError:

                                try:

                                    py_style_str = clean_str.replace("null", "None") \
                                        .replace("true", "True") \
                                        .replace("false", "False")

                                    parsed_obj = ast.literal_eval(py_style_str)
                                except (ValueError, SyntaxError):

                                    try:
                                        temp_obj = {}

                                        reasoning_match = re.search(r'"reasoning":\s*"(.*?)"\s*,\s*"tool_call"',
                                                                    clean_str, re.DOTALL)
                                        if reasoning_match:
                                            temp_obj['reasoning'] = reasoning_match.group(1)

                                        tool_name_match = re.search(r'"tool_name":\s*"(.*?)"', clean_str)

                                        params_match = re.search(r'"parameters":\s*(\{.*?\})\s*\}', clean_str,
                                                                 re.DOTALL)

                                        if tool_name_match:
                                            tool_call = {
                                                "tool_name": tool_name_match.group(1),
                                                "parameters": {}
                                            }

                                            if params_match:

                                                try:

                                                    p_str = params_match.group(1).replace('\\"', '"')
                                                    tool_call["parameters"] = json.loads(p_str)
                                                except:

                                                    tm_match = re.search(r'"target_mention":\s*"(.*?)"',
                                                                         params_match.group(1))
                                                    if tm_match:
                                                        tool_call["parameters"]["target_mention"] = tm_match.group(1)

                                            temp_obj['tool_call'] = tool_call
                                            parsed_obj = temp_obj
                                    except:
                                        pass

                        if parsed_obj and isinstance(parsed_obj, dict):
                            record['response'] = parsed_obj
                            fixed_count += 1
                        else:

                            pass

                    uid = str(record.get('id', '')).strip()
                    men = str(record.get('mentions', '')).strip()
                    if not men:
                        men = str(record.get('mentions', '')).strip()

                    unique_key = record_key(record)
                    unique_records_map[unique_key] = record

                except json.JSONDecodeError:
                    continue

            final_list = list(unique_records_map.values())

            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(final_list, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(final_list)} items to {target_file}")
            print(f"Successfully re-parsed {fixed_count} stringified responses into dicts.")

    def run(self):
        data = self.load_data()
        print(f"Processing {len(data)} samples...")

        realtime_input_file = self.paths['planner_input_running']
        realtime_result_file = self.paths['planner_running']

        processed_ids = set()

        processed_keys = set()
        if os.path.exists(realtime_result_file):
            print(f"[Resume] Found existing result log: {realtime_result_file}")
            print("[Resume] Scanning processed IDs...")
            with open(realtime_result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        if line.strip():
                            record = json.loads(line)

                            resp = record.get('response')

                            r_id = record.get('id')
                            if r_id is not None:

                                unique_key = record_key(record)
                                processed_keys.add(unique_key)

                    except json.JSONDecodeError:
                        continue
            print(f"[Resume] Already processed {len(processed_keys)} unique items.")

            if not os.path.exists(realtime_input_file):
                with open(realtime_input_file, 'w', encoding='utf-8') as f: pass

        else:
            print(f"[Start] No existing log file. Starting from scratch.")

            with open(realtime_result_file, 'w', encoding='utf-8') as f:
                pass
            with open(realtime_input_file, 'w', encoding='utf-8') as f:
                pass

        print(f"[Log] Real-time planner results will be saved to: {realtime_result_file}")

        print(f"Total samples: {len(data)}. Remaining: {len(data) - len(processed_keys)}")

        for sample_index, item in enumerate(tqdm(data, desc="Planner Loop")):

            current_men = str(item.get('mentions') or item.get('mention') or '').strip()
            current_key = record_key(item)

            if current_key in processed_keys:
                continue

            full_prompt, image_path, img_name, has_image = self.build_prompt(item)

            input_record = {
                "sample_index": sample_index,
                "id": item.get('id'),
                "mentions": current_men,
                "sentence": item.get('sentence'),
                "entity": item.get('entities'),
                "image_name": img_name,
                "answer": item.get('answer'),
                "answer_image": has_image
            }
            try:
                with open(realtime_input_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(input_record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Warning] Failed to write input log: {e}")

            img_base = None
            if has_image:
                try:
                    with open(image_path, "rb") as img_file:
                        img_base = base64.b64encode(img_file.read()).decode("utf-8")
                except Exception as e:
                    print(f"[Error] Failed to read image: {e}")
                    has_image = False

            content_list = []
            if has_image and img_base:
                content_list.append({"type": "image_url", "image_url": {"url": img_base}})
            content_list.append({"type": "text", "text": full_prompt})

            messages = [{"role": "user", "content": content_list}]

            parsed_response = None
            max_planner_retries = 10

            for attempt in range(max_planner_retries):
                try:
                    response = get_glm_response(
                        messages,
                        api_key=self.args.api_key,
                        model=self.args.policy_engine,
                        image_path=image_path,
                        temperature=self.args.policy_temperature,
                        max_tokens=self.args.policy_max_tokens
                    )
                    parsed_response = self.parse_llm_output(response)

                    response_str = str(parsed_response)

                    if "1301" in response_str or "sensitive content" in response_str.lower():
                        break

                    if isinstance(parsed_response, dict):
                        break
                    else:
                        print(f"[Planner Retry] ID {item.get('id')} produced non-dict output. Retrying...")

                except Exception as e:
                    print(f"[Planner Retry] API Error: {e}")
                    parsed_response = f"Error calling API: {str(e)}"

            result_record = {
                "sample_index": sample_index,
                "id": item.get('id'),
                "mentions": current_men,
                "sentence": item.get('sentence'),
                "response": parsed_response,
                "answer": item.get('answer')
            }

            try:
                with open(realtime_result_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Warning] Failed to write ID {item.get('id')}: {e}")

        self.convert_jsonl_to_json(realtime_result_file, self.paths['planner_out'], "Results")
        print("Planner Done!")

class PlanExecutor:
    def __init__(self, args):
        self.args = args
        self.paths = get_mode_paths(args)
        self.tool_engine = ToolEngine(args)
        self.image_root = os.path.join(args.data_root, args.image_dir)
        self.source_index_map = {}
        self.source_key_map = {}
        self.source_id_multi_map = {}
        self.source_id_map = self._load_source_data()

    def _load_source_data(self):
        source_path = self.paths['source_data']
        print(f"[Executor] Loading SOURCE data from {source_path}...")
        id_map = {}
        if os.path.exists(source_path):
            with open(source_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for sample_index, item in enumerate(data):
                    item_id = str(item.get('id'))
                    id_map[item_id] = item
                    self.source_index_map[str(sample_index)] = item
                    self.source_key_map[record_key(item)] = item
                    self.source_id_multi_map.setdefault(item_id, []).append(item)
            print(f"[Executor] Loaded {len(data)} source rows, {len(id_map)} unique ids.")
        else:
            print(f"[Error] Source data not found: {source_path}")
            sys.exit(1)
        return id_map

    def find_source_item(self, item):
        sample_index = item.get('sample_index')
        if sample_index is not None:
            source_item = self.source_index_map.get(str(sample_index))
            if source_item is not None:
                return source_item

        source_item = self.source_key_map.get(record_key(item))
        if source_item is not None:
            return source_item

        current_id = str(item.get('id'))
        candidates = self.source_id_multi_map.get(current_id, [])
        if candidates:
            mention = _norm_key_value(item.get('mentions') or item.get('mention'))
            answer = _norm_key_value(item.get('answer'))
            sentence = _norm_key_value(item.get('sentence'))
            for candidate in candidates:
                if (
                    _norm_key_value(candidate.get('mentions') or candidate.get('mention')) == mention
                    and _norm_key_value(candidate.get('answer')) == answer
                    and _norm_key_value(candidate.get('sentence')) == sentence
                ):
                    return candidate
            for candidate in candidates:
                if (
                    _norm_key_value(candidate.get('mentions') or candidate.get('mention')) == mention
                    and _norm_key_value(candidate.get('answer')) == answer
                ):
                    return candidate

        return self.source_id_map.get(current_id)

    def _extract_tool_call(self, item):
        response_data = item.get('response', {})
        if not isinstance(response_data, dict): return None
        tool_call = response_data.get('tool_call')
        if not tool_call:
            tool_call = response_data.get('response', {}).get('tool_call')
        return tool_call

    def _convert_jsonl_to_json(self, source_file, target_file, desc):
        import re
        import ast

        print(f"Converting {desc} (.jsonl) to final JSON (with Robust Parsing)...")
        if not os.path.exists(source_file):
            print(f"[Warning] {source_file} not found, skipping conversion.")
            return

        unique_records_map = {}
        fixed_count = 0

        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:

                    record = json.loads(line)

                    raw_response = record.get('response')

                    if isinstance(raw_response, str):
                        clean_str = raw_response.strip()
                        parsed_obj = None

                        match = re.search(r"```(?:json|JSON)?\s*(.*?)\s*```", clean_str, re.DOTALL)
                        if match:
                            clean_str = match.group(1)

                        try:
                            parsed_obj = json.loads(clean_str)
                        except json.JSONDecodeError:

                            try:

                                fixed_json_str = clean_str.replace('\\n', '\n').replace('\\"', '"')
                                parsed_obj = json.loads(fixed_json_str)
                            except json.JSONDecodeError:

                                try:

                                    py_style_str = clean_str.replace("null", "None") \
                                        .replace("true", "True") \
                                        .replace("false", "False")

                                    parsed_obj = ast.literal_eval(py_style_str)
                                except (ValueError, SyntaxError):

                                    try:
                                        temp_obj = {}

                                        reasoning_match = re.search(r'"reasoning":\s*"(.*?)"\s*,\s*"tool_call"',
                                                                    clean_str, re.DOTALL)
                                        if reasoning_match:
                                            temp_obj['reasoning'] = reasoning_match.group(1)

                                        tool_name_match = re.search(r'"tool_name":\s*"(.*?)"', clean_str)

                                        params_match = re.search(r'"parameters":\s*(\{.*?\})\s*\}', clean_str,
                                                                 re.DOTALL)

                                        if tool_name_match:
                                            tool_call = {
                                                "tool_name": tool_name_match.group(1),
                                                "parameters": {}
                                            }

                                            if params_match:

                                                try:

                                                    p_str = params_match.group(1).replace('\\"', '"')
                                                    tool_call["parameters"] = json.loads(p_str)
                                                except:

                                                    tm_match = re.search(r'"target_mention":\s*"(.*?)"',
                                                                         params_match.group(1))
                                                    if tm_match:
                                                        tool_call["parameters"]["target_mention"] = tm_match.group(1)

                                            temp_obj['tool_call'] = tool_call
                                            parsed_obj = temp_obj
                                    except:
                                        pass

                        if parsed_obj and isinstance(parsed_obj, dict):
                            record['response'] = parsed_obj
                            fixed_count += 1
                        else:

                            pass

                    uid = str(record.get('id', '')).strip()
                    men = str(record.get('mentions', '')).strip()
                    if not men:
                        men = str(record.get('mentions', '')).strip()

                    unique_key = record_key(record)
                    unique_records_map[unique_key] = record

                except json.JSONDecodeError:
                    continue

            final_list = list(unique_records_map.values())

            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(final_list, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(final_list)} items to {target_file}")
            print(f"Successfully re-parsed {fixed_count} stringified responses into dicts.")

    def run_execution(self):
        target_path = self.paths['planner_out']
        print(f"[Executor] Loading PLANNER output from {target_path}...")

        if not os.path.exists(target_path):
            alt_path = target_path.replace('.json', '.jsonl') if target_path.endswith('.json') else target_path.replace(
                '.jsonl', '.json')
            if os.path.exists(alt_path):
                target_path = alt_path
            else:
                return print(f"[Error] File not found: {target_path}")

        data_to_process = []
        with open(target_path, 'r', encoding='utf-8') as f:
            if target_path.endswith('.jsonl'):
                data_to_process = [json.loads(line) for line in f if line.strip()]
            else:
                data_to_process = json.load(f)

        if self.args.num_samples > 0:
            data_to_process = data_to_process[:self.args.num_samples]

        realtime_file = self.paths['executor_running']
        processed_keys = set()
        if os.path.exists(realtime_file):
            with open(realtime_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        processed_keys.add(record_key(record))
                    except:
                        continue
        else:
            open(realtime_file, 'w', encoding='utf-8').close()

        for item in tqdm(data_to_process, desc="Executor Loop"):
            current_id = str(item.get('id'))
            current_mention = item.get('mention') or item.get('mentions') or ""
            current_key = record_key(item)

            if current_key in processed_keys:
                continue

            source_item = self.find_source_item(item)
            if not source_item:
                item['tool_output'] = "Error: ID not found in source data."
                with open(realtime_file, 'a', encoding='utf-8') as f: f.write(
                    json.dumps(item, ensure_ascii=False) + "\n")
                continue

            raw_img_name = source_item.get('imgPath', "")
            real_img_path, exists, real_img_name = force_jpg_path(self.image_root, raw_img_name)

            tool_call_data = self._extract_tool_call(item)

            if isinstance(tool_call_data, dict):
                tool_name = tool_call_data.get('tool_name', 'Default_Case')
            elif isinstance(tool_call_data, str):
                tool_name = tool_call_data.strip()
            else:
                tool_name = "Default_Case"

            target_mention = source_item.get('mentions') or source_item.get('mention') or ""
            original_sentence = source_item.get('sentence', "")

            tool_input = {
                'id': current_id,
                'mention': target_mention,
                'sentence': original_sentence,
                'real_img_path': real_img_path if exists else None,
                'img_name': real_img_name
            }

            tool_res = {}
            if tool_name == "Default_Case":
                tool_res = self.tool_engine.run_default_case(tool_input)
            elif tool_name == "Text_Supplement":
                tool_res = self.tool_engine.run_text_supplement(tool_input)
            elif tool_name == "Symbolic_Solver":
                tool_res = self.tool_engine.run_symbolic_solver(tool_input)
            elif tool_name == "OCR_Augment":
                tool_res = self.tool_engine.run_ocr_augment(tool_input)
            elif tool_name == "Visual_Grounder":
                tool_res = self.tool_engine.run_visual_grounder(tool_input)
            else:
                tool_res = {"tool_output": f"Error: Unknown Tool '{tool_name}'"}

            item['tool_output'] = tool_res.get('tool_output', '')
            if 'grounded_image_path' in tool_res:

                item['processed_image'] = os.path.basename(tool_res['grounded_image_path'])

            try:
                with open(realtime_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Warning] Failed to write ID {current_id}: {e}")

        self._convert_jsonl_to_json(realtime_file, self.paths['final_out'],"Results")
        print("Executor execution finished.")

if __name__ == "__main__":
    args = parse_args()
    args = init_path_config(args)

    print(f"========== Dataset: {args.dataset} ==========")
    print(f"========== Mode: {args.mode} ==========")
    print(f"Source data: {get_mode_paths(args)['source_data']}")

    if args.stage in ('planner', 'all'):
        planner = Planner(args)
        planner.run()

    if args.stage in ('executor', 'all'):
        executor = PlanExecutor(args)
        executor.run_execution()

    print(f"========== {args.mode} mode completed ==========")
    print(f"Final output: {get_mode_paths(args)['final_out']}")
