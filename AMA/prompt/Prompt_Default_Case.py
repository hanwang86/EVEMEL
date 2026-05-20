prompt = """"
Role:
Your task is to identify a specific entity in a multimodal context, using both image and text, and provide a precise encyclopedic definition for disambiguation.

Input Data:
Entity Name: "{mention_name}"
Context: "{mention_context}"
Image: [Attached Image]

Task:
Describe: Based on the visual clues and the provided text context, generate a single-sentence description defining who or what the entity is.

Constraints:
- Focus only on the target entity itself. Do not describe the whole image, background, environment, or camera angle.
- The description must be factual, like the opening sentence of a Wikipedia entry.
- Use internal world knowledge to add distinguishing background information, such as nationality, occupation, location, or major achievements.
- Do not mention meaningless appearance details, such as clothing color. Focus on semantic identity or real-world status.
- The output must be purely objective. Do not use any wording that implies the existence of an image, such as “as seen in the image,” “shown in the picture,” or “the photo displays.”
- Use English.
- Keep the summary under 25 words.

Output Format:
[Your description]
"""
