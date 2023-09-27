import replicate
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN "))
st.set_page_config(page_title="Upscale", page_icon=":sparkles:")
st.sidebar.header("Upscale")
st.title("Upscale :sparkles:")
st.write("Tool che prende in input una immagine e diversi parametri per eseguire un up scale su di essa!")

@st.cache_resource
def generate_img(file_path, scale, isEnhanced):
    output = replicate.run(
        "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
        input={"image": open(file_path, "rb"),
               "scale": int(scale),
               "face_enhance" : isEnhanced}
    )
    st.write(output)
    st.image(output)
    
with st.form("upscale"):
    st.write("Parametri")
    image_path = st.file_uploader(label="Pick image", accept_multiple_files=False, type=["jpg", "jpeg", "png"])
    scale = st.slider(label="Scale", value=4.0, min_value=1.0, max_value=10.0, step=0.1)
    isEnhanced = st.checkbox(label="face_enhance")
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(image_path.read())
        st.write(f"Percorso del file caricato: {temp_file.name}")
        generate_img(temp_file.name, scale, isEnhanced)

