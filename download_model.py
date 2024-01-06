from transformers import AutoTokenizer
import transformers
import torch
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
MODEL = os.getenv("MODEL")
MODEL_PATH =  "./models/" + MODEL

tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_API_KEY)
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL,
    torch_dtype=torch.float16,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    token=HF_API_KEY
)

tokenizer.save_pretrained(MODEL_PATH)
pipeline.model.eval()
pipeline.model.save_pretrained(MODEL_PATH)
