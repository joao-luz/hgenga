from utils import save_txt

import re


class DescriptionGenerator:
    def __init__(self, class_task_desc, id2label, llm):
        self.class_task_desc = class_task_desc
        self.llm = llm
        self.id2label = id2label

    def get_descriptions(self, random_examples=[], positive_examples={}, negative_examples={}, similar_classes=None, return_llm_ouputs=True):
        random_examples_str = (
            f'\nHere are some examples of texts in the database:\n\n'
            f'```\n'
            f'{"\n\n".join(text for text in random_examples)}\n\n'
        ) if random_examples else ''
        
        def get_positive_examples_str(label):
            return (
                f'\n\nHere are some examples of texts correctly classified as "{self.id2label[label]}" based on a previously generated description:\n\n'
                f'```\n'
                f'{"\n\n".join(f"{e['text']}" for e in positive_examples[label])}\n'
                f'```\n\n'
            )

        def get_negative_examples_str(label):
            return (
                f'\n\nHere are some examples of texts wrongfuly classified as "{self.id2label[label]}" based on a previously generated description:\n\n'
                f'```\n'
                f'{"\n\n".join(f"{e['text']} - Classified as {e["pred"]}" for e in negative_examples[label])}\n'
                f'```\n\n'
            )
        
        def get_similar_classes_str(label):
            sim_classes = [self.id2label[int(c)] for c in similar_classes[label]]
            sim_examples = '\n\n'.join([f'{self.id2label[label]}: {positive_examples[int(c)][0]}' for c in similar_classes[label] if positive_examples.get(c)])
            return (
                f'\n\nClass "{self.id2label[label]}" is often confused with:\n'
                f'{"\n".join(sim_classes)}\n\n'
                f'Here are some examples:\n\n'
                f'{sim_examples}\n'
                f'In your definition, make sure to distinct "{self.id2label[label]}" from them.\n\n'
            )

        template = (
            f'You are a supporting agent for a text classification task. Here is the main task:\n\n'
            f'```\n'
            f'{self.class_task_desc}\n'
            f'```\n\n'
            f'You must generate a BRIEF text description for the "{{label}}" class to help the classification process, along with three '
            f'diverse examples that fit into this description. The examples should have a similar structure and LENGTH to other texts in the database. '
            f'The description should encompass known information about the class in order do help with misclassifications and should be brief.\n'
            f'{random_examples_str}{{positive_examples_str}}{{negative_examples_str}}{{similar_classes_str}}\n'
            f'Respond STRICTLY in the following JSON format:\n\n'
            f'{{{{\n'
            f'    "description": "class description",\n'
            f'    "example_1": "text for example_1"\n'
            f'    "example_2": "text for example_2"\n'
            f'    "example_3": "text for example_3"\n'
            f'}}}}'
        )

        prompts = []
        for i,l in self.id2label.items():
            positive_examples_str = get_positive_examples_str(i) if positive_examples.get(i) else ''
            negative_examples_str = get_negative_examples_str(i) if negative_examples.get(i) else ''
            similar_classes_str = get_similar_classes_str(i) if similar_classes is not None else ''

            prompts.append(template.format(label=l, 
                                        positive_examples_str=positive_examples_str, 
                                        negative_examples_str=negative_examples_str,
                                        similar_classes_str=similar_classes_str))

        responses = self.llm.invoke(prompts)

        def parse(response):
            desc_pat = r'"description": ["]*([^"}]+)["]*'
            example_pat = r'["example\_[0-9]+": "([^"]+)"[^"]*]*'

            try:
                examples = [a.strip() for a in re.findall(example_pat, response)]
                description = re.search(desc_pat, response).group(1).strip()
            except Exception as e:
                examples = []
                description = ''
            
            return {
                "description": description,
                "examples": examples
            }


        results = [parse(response) for response in responses]

        descriptions = {i: r['description'] for i,r in enumerate(results)}
        examples = {i: r['examples'] for i,r in enumerate(results)}

        if return_llm_ouputs:
            return descriptions, examples, (prompts, responses)
        
        return descriptions, examples