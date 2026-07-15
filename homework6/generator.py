import torch
from GeneratorTransformer import GeneratorTransformer


def chat():
    model = GeneratorTransformer(
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=4,
        dropout=0.1,
        max_len=64,
    )

    model.load_state_dict(torch.load("transformer_basics/checkpoints/best_model.pt")['model_state_dict'])
    model.to(torch.device('cuda'))

    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'quit':
            break

        response = model.generate(user_input, temperature=0.8, max_out_tokens=50)
        print(f"Бот: {response}")


if __name__ == '__main__':
    chat()
