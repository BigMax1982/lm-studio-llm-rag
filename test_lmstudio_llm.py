import requests
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse, CompletionResponseGen


class LMStudioLLM(CustomLLM):
    model: str
    api_base: str
    
    def __init__(self, model: str, api_base: str, **kwargs):
        super().__init__(model=model, api_base=api_base.rstrip("/"), **kwargs)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = requests.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get("temperature", 0.4)
            }
        )

        data = response.json()
        message = data["choices"][0]["message"]["content"]
        return CompletionResponse(text=message)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        # Streaming not implemented for simplicity
        response = self.complete(prompt, **kwargs)
        yield response

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)


# Instantiate LM Studio LLM
llm = LMStudioLLM(
    model="meta-llama-3.1-8b-instruct",
    api_base="http://localhost:1234/v1"
)

# Test call
response = llm.complete("Test: Explain in one sentence what WCAG Dragging Movements requires.")
print(response)
