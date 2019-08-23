from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIGPTLMHeadModel
import torch

class LanguageModel:
    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

     def score(self, text: str) -> float:
         """ Returns the sentence log probability score. """
         input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0) # Batch size 1
         outputs = self.model(input_ids, labels=input_ids)
         log_cross_entropy, _ = outputs[:2]
         log_probability = -log_cross_entropy.item()
         return log_probability
