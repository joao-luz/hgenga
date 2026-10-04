import torch
import re


class Labeler:
    def __init__(self, class_task_desc, id2label, llm):
        self.class_task_desc = class_task_desc
        self.llm = llm
        self.id2label = id2label
    
    def get_labels(self, descriptions, texts, prev_descriptions=None, prev_labels=None, return_llm_output=True):
        descriptions_str = (
            f'Look at the descriptions for the classes:\n\n'
            f'```\n'
            f'{"\n\n".join([f"{i}: {desc}" for i,desc in enumerate(descriptions)])}'
            f'```\n\n'
        ) if descriptions else ''
            

        def get_previous_str(id):
            if prev_labels is None or prev_descriptions is None:
                return ''
            
            prev_label_id = prev_labels[id]
            prev_label = self.id2label[prev_label_id]
            prev_class_desc = prev_descriptions[prev_label_id]
            return (
                f'Previously, the text was classified as "{prev_label}" under the following this PREVIOUS class description:\n\n'
                f'{prev_class_desc}\n\n'
            )

        template = (
            f'Task:\n\n'
            f'```\n'
            f'{self.class_task_desc}\n'
            f'```\n\n'
            f'Text: "{{text}}"\n\n'
            f'Look at the descriptions for the classes:\n\n'
            f'```\n'
            f'{descriptions_str}\n'
            f'```\n\n'
            f'{{previous_str}}'
            f'Based on the task, what is the class of the text and your reasoning?\n\n'
            f'Respond STRICTLY in the following JSON format and use ONLY the availabel label indices:\n\n'
            f'{{{{\n'
            f'    "reasoning": "brief reasoning",\n'
            f'    "label_id": "label index integer"\n'
            f'}}}}'
        )

        prompts = [template.format(text=text, previous_str=get_previous_str(i)) for i,text in enumerate(texts)]

        responses = self.llm.invoke(prompts, temperature=0.0)

        def parse(response):
            reasoning_pat = r'"reasoning": "([^"]+)"[^"]*]*'
            label_pat = r'"label_id": ["]*([^"}\s]+)["]*'

            try:
                reasoning = re.search(reasoning_pat, response).group(1).strip()
                label_id = re.search(label_pat, response).group(1).strip()
                label = int(label_id) if label_id else 0
                if label not in self.id2label:
                    label = 0
            except Exception as e:
                print(response)
                reasoning = ''
                label = 0
            
            return {
                "reasoning": reasoning,
                "label": label
            }

        answers = [parse(response.strip()) for response in responses]
        labels = torch.tensor([a['label'] for a in answers])
        reasonings = [a['reasoning'] for a in answers]

        if return_llm_output:
            return labels, reasonings, (prompts, responses)
        
        return labels, reasonings