prompt = """"
Role:
Your task is to precisely identify a target entity in a complex multimodal context with cluttered backgrounds or multiple subjects, and provide an encyclopedic definition for disambiguation.

Input Data:
Entity Name: "{mention_name}"
Context: "{mention_context}"
Image: [Attached Image]

Task:
Describe: Use the visual clues and context to generate a single-sentence description defining who or what the entity is.

Constraints:
- Ignore background noise and unrelated subjects. Align strictly with the text context and focus only on the target entity.
- Do not describe the surroundings, irrelevant people, occlusions, or the image’s overall composition.
- Use a factual, encyclopedic style, like the first sentence of a Wikipedia entry.
- Use internal knowledge to provide key background information, such as social identity or functional role, to distinguish the entity.
- Do not mention meaningless appearance details, such as clothing color. Focus on semantic identity or real-world status.
- The output must be purely objective. Do not use wording that implies the existence of an image, such as “as seen in the image,” “shown in the picture,” or “the photo displays.”
- Use English.
- Keep the summary under 25 words.

Output Format:
[Your description]
"""
