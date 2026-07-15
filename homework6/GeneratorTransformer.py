import torch
from transformer_basics.transformer import Transformer
from tokenizers import Tokenizer


class GeneratorTransformer(Transformer):
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 d_ff: int = 512,
                 num_layers: int = 6,
                 vocab_size: int = 1000,
                 pad_index: int = 1,
                 dropout: float = 0.1,
                 max_len: int = 64,
                 tokenizer: Tokenizer = None,
                 device: str = 'cuda', ):
        if tokenizer is None:
            tokenizer = Tokenizer.from_file("transformer_basics/mistral_tokenizer.json")
            tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])
            vocab_size = tokenizer.get_vocab_size()
            pad_index = tokenizer.token_to_id('<pad>')

        super().__init__(
            d_model,
            num_heads,
            d_ff,
            num_layers,
            vocab_size,
            pad_index,
            dropout,
            max_len,
            tokenizer,
            device
        )

    def generate(self, prompt, context_len=50, temperature=1.0, max_out_tokens=200):
        self.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids]).to(self.device)

            generated = input_ids.clone()

            for _ in range(max_out_tokens):
                outputs = self.predict(input_ids)
                next_token_logits = outputs[-1, :] / temperature

                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

                input_ids = generated[:, -context_len:]

                if next_token.item() == self.tokenizer.token_to_id('</s>'):
                    break
        return self.tokenizer.decode(generated[0].tolist())
