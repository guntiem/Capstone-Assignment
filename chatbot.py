import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class SimpleChatbot:
    def __init__(self):
        # Load pre-trained model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()  # Set the model to evaluation mode

    def generate_response(self, input_text):
        # Encode the input text and generate a response
        input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')

        with torch.no_grad():
            # Generate text
            output = self.model.generate(
                input_ids,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    chatbot = SimpleChatbot()
    print("Chatbot: Hello! I'm a simple chatbot. How can I help you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = chatbot.generate_response(user_input)
        print(f"Chatbot: {response}")
