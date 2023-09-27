import replicate
from replicate.exceptions import ModelError
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN "))
st.set_page_config(page_title="Restorantion", page_icon=":hammer:")
st.sidebar.header("Restorantion")
st.title("Restoration :hammer:")
st.write("Tool che prende in input una immagine e ne migliora la qualità (spesso non funziona).")

@st.cache_resource
def generate_img(file_path):
    try:
        output = replicate.run(
            "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
            input={"img": open(file_path, "rb"),
                "version": "v1.4",
                "scale" : 2}
        )
        st.write(output)
        st.image(output)
    except ModelError as e:
       st.warning(f"{e}")
       return
    
with st.form("Restoration"):
    st.write("Parametri")
    image_path = st.file_uploader(label="Pick image", accept_multiple_files=False, type=["jpg", "jpeg", "png"])
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(image_path.read())
        st.write(f"Percorso del file caricato: {temp_file.name}")
        generate_img(temp_file.name)
        

