from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIGPTLMHeadModel
import torch

class LanguageModel:
    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def score(self, text: str):
        indexed_tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])
        lm_logits, _ = self.model(tokens_tensor)
        return torch.gather(lm_logits, -1, tokens_tensor.unsqueeze(-1)).sum().item() / len(indexed_tokens)
