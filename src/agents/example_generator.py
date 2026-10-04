from collections import defaultdict
import random


class ExampleGenerator:
    def __init__(self, class_task_desc, id2label, llm):
        self.class_task_desc = class_task_desc
        self.llm = llm
        self.id2label = id2label

    def get_examples(self, descriptions, examples, n=1, temperature=0.7):
        template = (
            f"You are an expert information extraction agent supporting a text classification system.\n\n"
            f"Your goal is to generate a realistic text instance that belongs clearly and unambiguously "
            f"to a specific target class.\n\n"
            f"================ MAIN CLASSIFICATION TASK ================\n\n"
            f"```\n"
            f"{self.class_task_desc}\n"
            f"```\n\n"
            f"================ EXAMPLES FROM THE DATASET ================\n\n"
            f"```\n"
            f"{{examples}}\n"
            f"```\n\n"
            f"These examples illustrate the typical writing style, tone, structure, "
            f"and level of detail expected.\n\n"
            f"================ TARGET CLASS ================\n\n"
            f'Class: "{{label}}"\n\n'
            f"Description:\n\n"
            f"```\n"
            f"{{description}}\n"
            f"```\n\n"
            f"================ INSTRUCTIONS ================\n\n"
            f"1. Generate a new text that:\n"
            f"   - Clearly fits the target class.\n"
            f"   - Is consistent with the style and structure of the examples.\n"
            f"   - Contains strong signals that support the target classification.\n"
            f"2. Avoid ambiguity with other possible classes.\n"
            f"3. Do NOT explicitly mention the class name.\n"
            f"4. Do NOT explain your reasoning.\n"
            f"5. Do NOT include labels, prefixes, or commentary.\n\n"
            f"================ OUTPUT FORMAT (STRICT) ================\n\n"
            f"Return ONLY the generated text."
        )

        prompts = []
        classes = []
        for id,descriptions in descriptions.items():
            label = self.id2label[id]
            if isinstance(descriptions, list):
                for description in descriptions:
                    for _ in range(n):
                        examples_str = '\n\n'.join(random.sample(examples, k=5))
                        prompt = template.format(label=label, description=description, examples=examples_str)
                        prompts.append(prompt)
                        classes.append(id)
            else:
                description = descriptions
                for _ in range(n):
                    examples_str = '\n\n'.join(random.sample(examples, k=5))
                    prompt = template.format(label=label, description=description, examples=examples_str)
                    prompts.append(prompt)
                    classes.append(id)

        responses = self.llm.invoke(prompts, temperature=temperature)

        examples = defaultdict(list)
        for c,response in zip(classes, responses):
            examples[c].append(response)

        return examples