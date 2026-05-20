prompt=""""
Role:
Your task is to identify an abstract entity or broad concept in a multimodal context and provide a precise encyclopedic definition for disambiguation.

Input Data:
Entity Name: "{mention_name}"
Context: "{mention_context}"
Image: [Attached Image]

Task:
Describe: Generate a single-sentence description defining what the entity is.

Constraints:
- Treat the image only as a symbol or partial representation of the entity, such as a chart for “economy” or a street for “city.” Do not equate the entity with specific objects in the image.
- Infer the deeper meaning, cultural context, or overall atmosphere conveyed by the image, and combine it with the text context and world knowledge to define the entity.
- Focus only on the entity itself. Do not describe the literal image content.
- The output must be purely objective. Do not use any wording that implies the existence of an image, such as “as seen in the image,” “shown in the picture,” or “the photo displays.”
- Use English.
- Keep the summary under 25 words.

Output Format:
[Your description]
"""
