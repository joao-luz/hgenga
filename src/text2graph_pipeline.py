pipeline = [
    DocumentsToNodes(
        layer_name='documents'
    ),
    SentenceEmbedding(
        model='Qwen/Qwen3-VL-Embedding-2B',
        prompt=config['sentence_embedding_prompt']
    ),
    NodeGenerator(model='AxionML/Qwen3.5-9B-NVFP4', prompt=config['llm_node_gen_prompt'], n=5),
    KNNGraph(k=5),
    KMeansSampler(k=100, mask_name='to_label'),
    LLMLabeler(
        model='AxionML/Qwen3.5-9B-NVFP4', 
        prompt=config['llm_labeling_prompt'],
        label_map=config['label_map'],
        mask_name='to_label'),
    TemplateDescriptionGenerator(
        model='AxionML/Qwen3.5-9B-NVFP4', 
        prompt=config['llm_template_gen_prompt'], 
        k=5, 
        layer_name='descriptions'
    ),
    HeteroLabelPropagator(layers=['documents', 'descriptions'], epochs=200, lr=5e-4)
]

hetero_data = pipeline.run(documents=documents)