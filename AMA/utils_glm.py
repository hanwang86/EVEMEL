import os
import base64
import time
from openai import OpenAI

API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL_NAME = "glm-4-flash"

def get_client():
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_glm_response(messages, api_key, model="glm-4-flash", image_path=None, temperature=0, max_tokens=512, max_retries=10):
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"[API Error] Attempt {attempt + 1}/{max_retries} failed: {error_msg}")

            if "1301" in error_msg or "sensitive content" in error_msg.lower():
                return f"Error: {error_msg}"

            if attempt == max_retries - 1:
                return f"Error: {error_msg}"

            time.sleep(1)

    return "Error: Unknown failure"
