
from transformers import AutoTokenizer
import transformers
import torch
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL")
MODEL_PATH =  "./models/" + MODEL

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype=torch.float16,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

def generate_text(prompt):
  sequences = pipeline(
      prompt + '\n',
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      max_length=200,
  )
  for seq in sequences:
      print(f"Result: {seq['generated_text']}")

while True:
    print("Press Ctrl+C to exit.")
    prompt = input("Enter a prompt: ")
    generate_text(prompt)
