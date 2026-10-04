class Summarizer:
    def __init__(self, class_task_desc, id2label, llm):
        self.class_task_desc = class_task_desc
        self.llm = llm
        self.id2label = id2label

    def get_summarization(self, texts, return_llm_output=False):

        template = (
            'You are helping a text classification system.\n\n'

            'Classification task:\n'
            '---\n'
            '{task}\n'
            '---\n\n'

            'Input text:\n'
            '---\n'
            '{text}\n'
            '---\n\n'

            'Write a compressed task-oriented summary of the text.\n\n'

            'Requirements:\n'
            '- Preserve only information useful for classification\n'
            '- Keep class-discriminative details\n'
            '- Remove filler, redundancy, stylistic language, and unimportant details\n'
            '- Keep entities, domain terminology, symptoms, events, actions, keywords, or concepts that may help predict the class\n'
            '- Be concise but information-dense\n'
            '- Prefer noun phrases and compact factual statements\n'
            '- Do not explain your reasoning\n'
            '- Do not mention the task\n'
            '- Output only the summary\n\n'

            'Summary:'
        )

        prompts = [
            template.format(
                task=self.class_task_desc,
                text=text
            )
            for text in texts
        ]

        responses = self.llm.invoke(
            prompts,
            temperature=0.0
        )

        if return_llm_output:
            return prompts, responses

        return responses

    def get_features(
        self,
        extracted_features,
        return_llm_output=False
    ):

        extracted_features_str = '\n'.join(
            f'- {feature}'
            for feature in extracted_features
        )

        prompt = (
            'You are designing features for a text classification system.\n\n'

            'Classification task:\n'
            '---\n'
            f'{self.class_task_desc}\n'
            '---\n\n'

            'Below are compressed summaries or extracted signals from texts:\n\n'

            'Features:\n'
            '---\n'
            f'{extracted_features_str}\n'
            '---\n\n'

            'Identify the most discriminative semantic features for distinguishing classes.\n\n'

            'Good features:\n'
            '- are predictive of class membership\n'
            '- generalize across many examples\n'
            '- are semantically meaningful\n'
            '- are not overly specific or noisy\n'
            '- help separate confusing classes\n\n'

            'Avoid:\n'
            '- generic language features\n'
            '- formatting artifacts\n'
            '- overly broad concepts\n'
            '- redundant features\n'
            '- features tied to a single example\n\n'

            'Return exactly 10 feature names.\n\n'

            'Output rules:\n'
            '- one feature per line\n'
            '- short feature names only\n'
            '- no numbering\n'
            '- no explanations\n'
            '- no extra text'
        )

        response = self.llm.invoke(
            prompt,
            temperature=0.0
        )

        features = [
            feature.strip()
            for feature in response.split('\n')
            if feature.strip()
        ]

        if return_llm_output:
            return prompt, response, features

        return features