import base64
import streamlit as st

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

st.title("LLaVA Demo")

@st.cache_resource
def load_model():
    chat_handler = Llava15ChatHandler(clip_model_path="model/mmproj-model-f16.gguf")
    llm = Llama(
    model_path="model/ggml-model-q4_k.gguf",
    chat_handler=chat_handler,
    n_ctx=2048, # n_ctx should be increased to accomodate the image embedding
    logits_all=True,# needed to make llava work
    )
    return llm

def get_base64(data):
    b64data = base64.b64encode(data.read()).decode("utf-8")
    return f"data:image/png;base64,{b64data}"

def generate(image, prompt):
    llm = load_model()
    streamer = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "あなたは完璧に画像を日本で説明するアシスタントです"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": get_base64(image)}},
                        {"type" : "text", "text": f"日本語で回答してください。{prompt}"}
                    ]
                }
        ],
        stream = True
    )
    partial_message = ""
    for msg in streamer:
        delta_msg = msg["choices"][0]["delta"].get("content")
        if delta_msg:
            partial_message += delta_msg
        yield partial_message

def main():
    image = st.file_uploader("upload image", type=["png", "jpg"])
    if image:
        st.image(image)

    if prompt := st.chat_input("モデルに聞きたいことを書いてください"):
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking"):
                placeholder = st.empty()
                for msg in generate(image, prompt):
                    placeholder.markdown(msg)

if __name__ == "__main__":
    main()
