from collections import defaultdict
import random


class ClassDescriptionGenerator:
    def __init__(self, class_task_desc, id2label, llm):
        self.class_task_desc = class_task_desc
        self.id2label = id2label
        self.llm = llm

    def get_descriptions(self, examples_by_class, sample_size=20, temperature=0.7):
        template = (
            "You are an expert taxonomy designer for a text classification system.\n\n"

            "Your task is to infer the defining characteristics of a class "
            "from a collection of example documents.\n\n"

            "================ CLASSIFICATION TASK ================\n\n"
            "```\n"
            "{task_desc}\n"
            "```\n\n"

            "================ EXAMPLE DOCUMENTS ================\n\n"
            "The following documents belong to the SAME class:\n\n"

            "```\n"
            "{examples}\n"
            "```\n\n"

            "================ OBJECTIVE ================\n\n"

            "Analyze the documents and produce a concise description of the "
            "underlying class.\n\n"

            "The description should capture:\n"
            "- recurring topics\n"
            "- common terminology\n"
            "- characteristic entities\n"
            "- typical events\n"
            "- writing patterns\n"
            "- semantic signals that distinguish these documents from others\n\n"

            "Do NOT:\n"
            "- summarize individual documents\n"
            "- mention document counts\n"
            "- refer to examples explicitly\n"
            "- invent information not supported by the texts\n\n"

            "The description should help a classifier identify future "
            "documents belonging to the same class.\n\n"

            "================ OUTPUT FORMAT ================\n\n"

            "Return ONLY the class description."
        )

        prompts = []
        class_ids = []

        print(sample_size)

        for class_id, texts in examples_by_class.items():

            sampled = random.sample(
                texts,
                k=min(sample_size, len(texts))
            )

            examples_str = "\n\n".join(sampled)

            prompt = template.format(
                task_desc=self.class_task_desc,
                examples=examples_str,
            )

            prompts.append(prompt)
            class_ids.append(class_id)

        responses = self.llm.invoke(
            prompts,
            temperature=temperature,
        )

        descriptions = {}

        for class_id, response in zip(class_ids, responses):
            descriptions[class_id] = (response.strip())

        return descriptions