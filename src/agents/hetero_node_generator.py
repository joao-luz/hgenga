from collections import defaultdict
import random


class HeteroNodeGenerator:
    def __init__(self, class_task_desc, id2label, llm):
        self.class_task_desc = class_task_desc
        self.llm = llm
        self.id2label = id2label

    def get_nodes(
        self,
        descriptions,
        examples,
        n=3,
        examples_per_prompt=5,
        temperature=0.7,
    ):

        template = (
            "You are an expert graph construction agent for heterogeneous "
            "graph neural networks used in text classification.\n\n"

            "Your task is to generate SEMANTIC ANCHOR NODES that will be "
            "embedded using a text embedding model and connected to document "
            "nodes through cosine similarity.\n\n"

            "IMPORTANT:\n"
            "These nodes are NOT labels or summaries.\n"
            "They must be written so that their embeddings naturally become "
            "similar to real documents from the target class.\n\n"

            "================ MAIN TASK ================\n\n"
            "```\n"
            "{task_desc}\n"
            "```\n\n"

            "================ REAL DATASET EXAMPLES ================\n\n"
            "```\n"
            "{examples}\n"
            "```\n\n"

            "================ TARGET CLASS ================\n\n"
            'Class: "{label}"\n\n'

            "Class Description:\n\n"
            "```\n"
            "{description}\n"
            "```\n\n"

            "================ OBJECTIVE ================\n\n"

            "Generate ONE short semantic prototype text that could serve as a "
            "heterogeneous graph node.\n\n"

            "The generated node should:\n"
            "- resemble phrases or passages likely to appear in real documents\n"
            "- contain terminology strongly associated with the target class\n"
            "- contain discriminative semantic signals\n"
            "- share vocabulary with documents from this class\n"
            "- be useful for embedding similarity retrieval\n"
            "- help connect semantically related documents together\n\n"

            "The node should NOT:\n"
            "- be abstract or overly generic\n"
            "- be a label name\n"
            "- summarize the class at a high level\n"
            "- explain anything\n"
            "- mention classification\n\n"

            "GOOD examples:\n"
            "- 'progressive shortness of breath with bilateral pulmonary infiltrates'\n"
            "- 'unauthorized wire transfers and falsified accounting records'\n"
            "- 'recursive GPU kernel optimization for sparse matrix multiplication'\n"
            "- 'persistent abdominal pain after laparoscopic bowel resection'\n"
            "- 'customer complaints regarding delayed shipment and refund processing'\n\n"

            "BAD examples:\n"
            "- 'medical condition'\n"
            "- 'financial crime'\n"
            "- 'computer science topic'\n"
            "- 'important clinical finding'\n"
            "- 'this document discusses fraud'\n\n"

            "IMPORTANT:\n"
            "The text should maximize semantic similarity with real documents "
            "from the class when encoded with a sentence embedding model.\n\n"

            "================ OUTPUT FORMAT ================\n\n"
            "Return ONLY the generated node text.\n"
            "No explanations.\n"
            "No bullet points.\n"
            "No prefixes."
        )

        prompts = []
        classes = []

        for class_id, class_descriptions in descriptions.items():

            label = self.id2label[class_id]

            if not isinstance(class_descriptions, list):
                class_descriptions = [class_descriptions]

            for description in class_descriptions:

                for _ in range(n):

                    sampled_examples = random.sample(
                        examples,
                        k=min(examples_per_prompt, len(examples))
                    )

                    examples_str = "\n\n".join(sampled_examples)

                    prompt = template.format(
                        task_desc=self.class_task_desc,
                        examples=examples_str,
                        label=label,
                        description=description,
                    )

                    prompts.append(prompt)
                    classes.append(class_id)

        responses = self.llm.invoke(
            prompts,
            temperature=temperature
        )

        generated_nodes = defaultdict(list)

        for c, response in zip(classes, responses):
            generated_nodes[c].append(response.strip())

        return generated_nodes