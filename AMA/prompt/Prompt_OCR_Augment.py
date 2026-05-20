prompt=""""
Role:
Your task is to identify the target entity using context, extracted OCR text, and visual clues, then provide a precise text-only encyclopedic definition for entity linking.

Input Data:
Entity Name: "{mention_name}"
Context: "{mention_context}"
Raw OCR Text: "{ocr_extracted_text}"
Image: [Attached Image]

Task:
Entity Definition: Treat OCR text and visual features, such as logos or distinctive typography, only as clues to confirm the entity’s real-world identity. Then use world knowledge to generate a single objective sentence defining what or who the Entity Name refers to.

Constraints:
- Focus only on the input Entity Name. OCR text is only supporting evidence. For example, if the entity is "TED", define TED itself, not a specific TEDx event found in OCR.
- The output must be a purely objective factual definition. Do not describe the image scene, text carrier, layout, or position.
- Filter OCR noise such as typos, fragments, and irrelevant background text.
- Use English.
- Keep the summary under 25 words.

Output Format:
[Your description]
"""
