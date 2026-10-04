import vllm

import gc
import ray
import torch

class LLM():
    def __init__(self, model_name, load_model=False, **vllm_args):
        self.model_name = model_name
        self.loaded = False
        self.model = None
        self.default_chat_kwargs = {}
        self.default_vllm_args = {'max_model_len': 4096, 'gpu_memory_utilization': 0.7} | vllm_args

        if load_model:
            self.load_model(**vllm_args)

    
    def unload_model(self):
        if not self.loaded:
            return
        
        try:
            del self.model
        except Exception as e:
            print(f"Failed to unload model: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()
            
        self.model = None
        self.loaded = False


    def __del__(self):
        self.unload_model()


    def load_model(self, **vllm_args):
        if self.loaded:
            return
        
        vllm_args = self.default_vllm_args | vllm_args
        self.model = vllm.LLM(
            model=self.model_name, 
            trust_remote_code=True,
            language_model_only=True,
            **vllm_args
        )

        self.loaded = True
    

    def set_default_chat_kwargs(self, chat_kwargs):
        self.default_chat_kwargs = chat_kwargs

    
    def invoke(self, prompts, temperature=0.7, sampling_kwargs={}, chat_kwargs={}):
        if not self.loaded:
            self.load_model()

        single_prompt = False
        if not isinstance(prompts, list):
            prompts = [prompts]
            single_prompt = True

        chats = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
        sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=2048, **sampling_kwargs)

        chat_kwargs = chat_kwargs or self.default_chat_kwargs
        outputs = self.model.chat(chats, sampling_params, **chat_kwargs)
        generated_texts = [output.outputs[0].text for output in outputs]

        return generated_texts if not single_prompt else generated_texts[0]
    
    
    def embed(self, prompts):
        outputs = self.model.encode(prompts, pooling_task='embed')

        embs = torch.stack([o.outputs.data for o in outputs])

        return embs
