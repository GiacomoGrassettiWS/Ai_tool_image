import replicate
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN "))
st.set_page_config(page_title="Restorantion", page_icon=":star-struck:")
st.sidebar.header("Restorantion")
st.title("Stable diffusion :star-struck:")
st.write("Tool che permette di creare immagini da prompt, usando diversi meccanismi e parametri tramite il modello 'Stable diffusion'")

@st.cache_resource
def generate_img(prompt, height, width, negativePrompt, numOuput, interferenceStep, guidanceScale, scheduler, seed):
    output = replicate.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={"prompt": prompt,
               "height" : height,
               "width" : width,
               "negative_prompt" : negativePrompt,
               "num_outputs" : numOuput,
               "num_inference_steps" : interferenceStep,
               "guidance_scale" : guidanceScale,
               "scheduler" : scheduler,
               "seed" : seed
               }
        )
    print(output)
    return output
    
with st.form("Restoration"):
    st.write("Parametri")
    
    prompt = st.text_input(label="Prompt")
    negativePrompt = st.text_input(label="Negative prompt")
    height = st.slider(label="Height", value=768, min_value=64, max_value=1024, step=64)
    width = st.slider(label="Width", value=768, min_value=64, max_value=1024, step=64)
    numOuput = st.slider(label="N. output", value=1, min_value=1, max_value=4, step=1)
    interferenceStep = st.slider(label="N. step", value=50, min_value=1, max_value=500, step=1)   
    guidanceScale = st.slider(label="N. step", value=7.5, min_value=1.0, max_value=20.0, step=0.1)
    scheduler = st.selectbox(label="Scheduler", options=("DDIM", "K_EULER", "DPMSolverMultistep", "K_EULER_ANCESTRAL", "PNDM", "KLMS"), index=3)
    seed = st.number_input(label="Seed", placeholder="leave 0 for random seed", min_value=0, step=1, value=0)
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        outputs = generate_img(prompt, height, width, negativePrompt, numOuput, interferenceStep, guidanceScale, scheduler, seed)
        for img in outputs:
            st.write(img)
            st.image(img) 