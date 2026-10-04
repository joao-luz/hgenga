import re


class DescriptionMatcher:
    def __init__(self, class_task_desc, id2label, llm):
        self.class_task_desc = class_task_desc
        self.id2label = id2label
        self.llm = llm

    def match_descriptions(self, texts, labels, descriptions, temperature=0.0, return_llm_output=False):
        template = (
            f"You are an expert text classification analyst.\n\n"

            f"Your task is to determine which candidate descriptions "
            f"correctly characterize a document.\n\n"

            f"================ MAIN CLASSIFICATION TASK ================\n\n"

            f"```\n"
            f"{self.class_task_desc}\n"
            f"```\n\n"

            f"================ TARGET LABEL ================\n\n"

            f"Label: \"{{label}}\"\n\n"

            f"================ DOCUMENT ================\n\n"

            f"```text\n"
            f"{{text}}\n"
            f"```\n\n"

            f"================ CANDIDATE DESCRIPTIONS ================\n\n"

            f"{{descriptions}}\n\n"

            f"================ INSTRUCTIONS ================\n\n"

            f"Select the descriptions that clearly match the document.\n\n"

            f"A description matches if it captures important characteristics, "
            f"themes, intent, language patterns, structure, or content present "
            f"in the document.\n\n"

            f"Only select descriptions strongly supported by the text.\n\n"

            f"Avoid:\n"
            f"- Selecting weak or ambiguous matches.\n"
            f"- Selecting descriptions contradicted by the document.\n"
            f"- Over-selecting descriptions.\n\n"

            f"================ OUTPUT FORMAT (STRICT) ================\n\n"

            f"Return ONLY a numbered list of matching description indices.\n\n"

            f"Example:\n"
            f"1\n"
            f"3\n"
            f"5\n\n"

            f"If none match, return:\n"
            f"NONE\n"
        )

        def parse(response):
            response = response.strip()

            if response.upper() == "NONE":
                return []

            matches = re.findall(r'^\s*(\d+)\s*$', response, flags=re.MULTILINE)

            if not matches:
                raise ValueError(
                    f"Could not parse matching description indices.\n\n"
                    f"Response:\n{response}"
                )

            return [int(x) - 1 for x in matches]

        prompts = []

        for text, label_id in zip(texts, labels):
            label_name = self.id2label[label_id]
            label_descriptions = descriptions[label_id]

            descriptions_str = "\n".join(
                f"{i+1}. {desc}"
                for i, desc in enumerate(label_descriptions)
            )

            prompt = template.format(
                label=label_name,
                text=text,
                descriptions=descriptions_str
            )

            prompts.append(prompt)

        responses = self.llm.invoke(prompts, temperature=temperature)

        parsed = [parse(r) for r in responses]

        if return_llm_output:
            return parsed, responses

        return parsed