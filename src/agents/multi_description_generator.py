import re
import random


class MultiDescriptionsGenerator:
    def __init__(self, class_task_desc, id2label, llm):
        self.class_task_desc = class_task_desc
        self.llm = llm
        self.id2label = id2label

    def get_descriptions(self, examples_by_class=None, sample_size=0, temperature=0.7):
        template = (
            f"You are a taxonomy design expert for a text classification system.\n\n"
            f"Your task is to generate multiple independent class descriptions "
            f"that could be used to determine whether a document belongs to a given class.\n\n"
            f"================ MAIN CLASSIFICATION TASK ================\n\n"
            f"```\n"
            f"{self.class_task_desc}\n"
            f"```\n\n"
            f"================ TARGET CLASS ================\n\n"
            f"Class: \"{{label}}\"\n\n"
            f"{{examples_str}}"
            f"================ INSTRUCTIONS ================\n\n"
            f"Generate a list of distinct descriptions that a document could be matched against "
            f"to decide if it belongs to this class.\n\n"
            f"Each description should:\n"
            f"- Clearly characterize documents that belong to this class.\n"
            f"- Emphasize different defining aspects (e.g., themes, intent, language patterns, structure, typical content).\n"
            f"- Highlight signals that help distinguish this class from others.\n"
            f"- Be self-contained and usable as a standalone matching criterion.\n"
            f"- Not explicitly mention the class name inside the description.\n\n"
            f"Avoid:\n"
            f"- Rephrasing the class name without adding substance.\n"
            f"- Very short or vague descriptions.\n"
            f"- Redundant descriptions that express essentially the same criteria.\n"
            f"- Introducing concepts unrelated to the main task.\n\n"
            f"================ OUTPUT FORMAT (STRICT) ================\n\n"
            f"Return a numbered list.\n\n"
            f"1. <DESCRIPTION>\n"
            f"2. <DESCRIPTION>\n"
            f"3. <DESCRIPTION>\n"
            f"...\n"
        )

        def parse(response):
            pattern = r'\n?\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|\Z)'

            matches = list(re.finditer(pattern, response))

            if not matches:
                raise ValueError("No valid numbered descriptions found.")

            descriptions = []
            for match in matches:
                text = match.group(2).strip()

                descriptions.append(text)

            return descriptions

        prompts = []
        for id,label in self.id2label.items():
            examples_str = ''
            if examples_by_class:
                examples_str = '\n\n'.join(random.sample(examples_by_class[id], k=min(sample_size, len(examples_by_class[id]))))
                examples_str = (
                    f'================ CLASS EXAMPLES ================\n\n'
                    f'{examples_str}\n\n'
                )
            prompt = template.format(label=label, examples_str=examples_str)
            prompts.append(prompt)

        responses = self.llm.invoke(prompts, temperature=temperature)

        descriptions = {i: parse(response) for i,response in enumerate(responses)}

        return descriptions