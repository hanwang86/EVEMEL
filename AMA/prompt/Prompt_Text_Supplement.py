prompt = """"
Role:
Your task is to provide an objective encyclopedic or typical visual description of the target entity based only on text context, helping downstream entity linking when no image is available.

Input Data:
Entity Name: "{mention_name}"
Context: "{mention_context}"

Task:
Infer the real identity of the Entity Name in the sentence, then use world knowledge to provide one objective encyclopedic definition. Emphasize typical static features or visual appearance.

Constraints:
- The output must be a general encyclopedic definition. For example, if the entity is a college sports team, describe it as a team representing a higher education institution, usually wearing uniforms with the university logo and colors.
- Do not describe specific events, actions, or states from the current context.
- Ignore irrelevant entities and details. Focus only on the essential nature of the Entity Name.
- Use English.
- Keep the summary under 30 words.

Output Format:
[Your single-sentence description]
"""
