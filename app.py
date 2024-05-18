import os
import base64
import streamlit as st

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from llama_cpp.llama_chat_format import MoondreamChatHandler

from utils.mylogger import getLogger, set_logger

set_logger()
logger = getLogger(__file__)

MAX_WINDOW = 4

st.title("LLaVA Demo")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "image" not in st.session_state:
    st.session_state.image = None

MODEL_DIR = "model"
MODEL_LIST = {
    "llava-v1.5-7b": {
        "mmproj": {
            "repo_id": "mys/ggml_llava-v1.5-7b",
            "filename": "mmproj-model-f16.gguf"
        },
        "text-model": {
            "repo_id": "mys/ggml_llava-v1.5-7b",
            "filename": "ggml-model-q4_k.gguf"
        }
    },
    "bakllava-1-7b": {
        "mmproj": {
            "repo_id": "mys/ggml_bakllava-1",
            "filename": "mmproj-model-f16.gguf"
        },
        "text-model": {
            "repo_id": "mys/ggml_bakllava-1",
            "filename": "ggml-model-q4_k.gguf"
        }
    },
    "moondream2": {
        "mmproj": {
            "repo_id": "vikhyatk/moondream2",
            "filename": "moondream2-mmproj-f16.gguf"
        },
        "text-model": {
            "repo_id": "vikhyatk/moondream2",
            "filename": "moondream2-text-model-f16.gguf"
        }
    },
    "llava-v1.5-13b": {
        "mmproj": {
            "repo_id": "mys/ggml_llava-v1.5-13b",
            "filename": "mmproj-model-f16.gguf"
        },
        "text-model": {
            "repo_id": "mys/ggml_llava-v1.5-13b",
            "filename": "ggml-model-q4_k.gguf"
        }
    },
    "llava-v1.6-34b": {
        "mmproj": {
            "repo_id": "cjpais/llava-v1.6-34B-gguf",
            "filename": "mmproj-model-f16.gguf"
        },
        "text-model": {
            "repo_id": "cjpais/llava-v1.6-34B-gguf",
            "filename": "llava-v1.6-34b.Q4_K_M.gguf"
        }
    },
}

@st.cache_resource
def load_model(model_name):
    model_info = MODEL_LIST[model_name]
    if os.path.exists(os.path.join(MODEL_DIR, model_info["mmproj"]["filename"])) and os.path.exists(os.path.join(MODEL_DIR, model_info["text-model"]["filename"])):
        llm = load_model_from_local(model_name)
    else:
        llm = load_model_from_remote(model_name)
    return llm

def load_model_from_local(model_name):
    model_info = MODEL_LIST[model_name]
    mmproj = model_info["mmproj"]
    text_model = model_info["text-model"]
    if model_name == "moondream2":
        chat_handler = MoondreamChatHandler(
            clip_model_path=os.path.join(MODEL_DIR, mmproj["filename"])
        )
    else:
        chat_handler = Llava15ChatHandler(clip_model_path=os.path.join(MODEL_DIR, mmproj["filename"]))

    llm = Llama(
        model_path=os.path.join(MODEL_DIR, text_model["filename"]),
        chat_handler=chat_handler,
        n_ctx=2048, # n_ctx should be increased to accomodate the image embedding
        logits_all=True, # needed to make llava work
        temperature=0.5,
        repeat_penalty=1.1
    )
    return llm

def load_model_from_remote(model_name):
    model_info = MODEL_LIST[model_name]
    mmproj = model_info["mmproj"]
    text_model = model_info["text-model"]
    if model_name == "moondream2":
        chat_handler = MoondreamChatHandler.from_pretrained(
            repo_id=mmproj["repo_id"],
            filename=mmproj["filename"]
        )
    else:
        chat_handler = Llava15ChatHandler.from_pretrained(
            repo_id=mmproj["repo_id"],
            filename=mmproj["filename"]
        )
    llm = Llama.from_pretrained(
        repo_id=text_model["repo_id"],
        filename=text_model["filename"],
        chat_handler=chat_handler,
        n_ctx=2048, # n_ctx should be increased to accomodate the image embedding
        logits_all=True, # needed to make llava work
        temperature=0.5,
        repeat_penalty=1.1
    )
    return llm

def get_base64(data):
    b64data = base64.b64encode(data.read()).decode("utf-8")
    return f"data:image/png;base64,{b64data}"

def generate(image, prompt, model_name):
    llm = load_model(model_name)
    if len(st.session_state.messages) == 0:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "あなたは完璧に画像を日本語で説明するアシスタントです"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": get_base64(image)
                        }
                    },
                    {
                        "type" : "text",
                        "text": f"日本語で回答してください。{prompt}"
                    }
                ]
            }
        ]
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": get_base64(image)
                        }
                    },
                    {
                        "type" : "text",
                        "text": f"日本語で回答してください。{prompt}"
                    }
                ]
            }
        )
    streamer = llm.create_chat_completion(
        messages = st.session_state.messages[-MAX_WINDOW:],
        stop=["SYSTEM:", "USER:", "ASSISTANT:"],
        stream = True
    )
    partial_message = ""
    for msg in streamer:
        delta_msg = msg["choices"][0]["delta"].get("content")
        if delta_msg:
            partial_message += delta_msg
        yield partial_message

def main():
    model_name = st.selectbox("使用するモデルを選んでください", MODEL_LIST.keys())
    image = st.file_uploader("upload image", type=["png", "jpg"])
    if image:
        st.image(image)
        if image != st.session_state.image:
            st.session_state.messages = []
            st.session_state.image = image

    for message in st.session_state.messages:
        if not message["role"] in ["assistant", "user"]:
            continue
        with st.chat_message(message["role"]):
            if type(message.get("content")) is str:
                st.markdown(message["content"])
            else:
                st.markdown(message["content"][1]["text"])

    if prompt := st.chat_input("モデルに聞きたいことを書いてください"):
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking"):
                placeholder = st.empty()
                for msg in generate(image, prompt, model_name):
                    placeholder.markdown(msg)
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": msg
                        }
                    )
        logger.info([{"role": message["role"], "content": message["content"] if type(message.get("content")) is str else message["content"][1]["text"]} for message in st.session_state.messages])

if __name__ == "__main__":
    main()
