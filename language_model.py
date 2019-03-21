from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch


class LanguageModel:
    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        config = GPT2Config()
        self.model = GPT2LMHeadModel(config)
        self.model.eval()

    def score(self, text):
        indexed_tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])
        lm_logits, _ = self.model(tokens_tensor)
        return torch.gather(lm_logits, -1, tokens_tensor.unsqueeze(-1)).sum().item()
