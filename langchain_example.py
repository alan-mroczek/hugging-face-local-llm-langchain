
from transformers import AutoTokenizer, TextIteratorStreamer
import transformers
import threading
import torch
import os
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage

from dotenv import load_dotenv

load_dotenv()

# TODO: make this better aligned with meta examples, it's not perfect
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

MODEL = os.getenv("MODEL")
MODEL_PATH =  "./models/" + MODEL

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=2)
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype=torch.float16,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    streamer=streamer,
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={"temperature": 0})


# TODO: refine this function, figure out if you want to use predict_messages
# do we need langchain.prompts.chat ?
def messages_to_llama2_prompt(messages):
    result = f"{B_SYS} {messages[0].content} {E_SYS}"
    for message in messages[1:]:
        if isinstance(message, HumanMessage):
            result += f"{B_INST} {message.content} {E_INST}"
        elif isinstance(message, AIMessage):
            result += f" {message.content} "
        else:
            raise ValueError("Unknown message type.")
        
    return result

print("Press Ctrl+C to exit.")

input_subject = input("Provide university subject: ")

template = "Act as an professor that teaches {subject}. Be kind and helpful."

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(template),
        HumanMessage(content="Hello!"),
        AIMessage(content="Welcome my student!"),
    ]
)

messages = chat_prompt.format_messages(
    subject=input_subject,
)

def predict_messages(messages):
    return llm.predict(messages_to_llama2_prompt(messages))

while True:
    print("=====================================")
    input_message = input("You: ")

    messages.append(HumanMessage(content=input_message))
    
    print("=====================================")
    print(f"Professor: ", end="")

    result = ""

    thread = threading.Thread(target=predict_messages, args=(messages,))
    thread.start()

    # streaming line by line
    for new_text in streamer:
        text = new_text.replace("</s>", "")
        print(text, end="")
        result += text

    print()

    messages.append(AIMessage(content=result))
