# Hugging Face Local LLM + LangChain

## Description

!WORK IN PROGRESS!

This is a project to explore and showcase using LLM models from Hugging Face locally in LangChain. For my example I am using Meta's [LLama 2 7B chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model. In order to use this model you will need to get an access from Meta and accept their terms and conditions. To use other models you will need to edit .env file and change the model name, just make sure it is a chat model in Hugging Face format based on Llama 2.

eg. `meta-llama/Llama-2-7b-chat-hf` not `meta-llama/Llama-2-7b-chat`

Few examples of models that should work:

- `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf` etc.
- `NousResearch/Llama-2-7b-chat-hf`, `NousResearch/Llama-2-13b-chat-hf` etc.

## Pre-requisites

1. API key from Hugging Face in the environment file `.env`

    ```bash
    HUGGING_FACE_API_KEY=...
    ```

2. CUDA enabled GPU and CUDA toolkit installed (for GPU acceleration)
    - make sure that you're using a strong GPU (eg. RTX 4090, A100) for fast inference

## Usage

1. Init venv and install dependencies

    ```bash
    python -m venv .venv
    source ./.venv/bin/activate
    pip install -r requirements.txt
    ```

2. Run the script (this will download the model and save it to `./models` - not really necessary - without it, it would download model to cache and could be used directly from there)

    I am saving it to `./models` only to make it more transparent and avoid "magic".

    ```bash
    python ./download_model.py
    ```

    It will take a while to download the model, it is ~13GB in size.

3. Running pipeline directly or via LangChain

    I have two examples - `pipeline.py` and `langchain.py`. First one is a simple pipeline that will generate a response to a given input. Second one is an example of using this model in LangChain. Both run fully offline using only local resources and are stateless. You can use them as a starting point for your own project.

    Loading model takes a while (~2 minutes on my setup) - I am not entirely sure why it's much slower compared to running model direcly from Llama 2 repo instead of Hugging Face - work in progress. Once you load the model, inference is very fast.

    ```bash
    python ./pipeline.py
    ```

    ```bash
    python ./langchain_example.py
    ```

