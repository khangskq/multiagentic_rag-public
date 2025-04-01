from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ModelEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 500,
        temperature: float = 0.7
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
