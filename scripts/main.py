import streamlit as st

st.set_page_config(
    page_title="Meta-Ent Demo Site",
    page_icon="/home/td/CI/CI_METAVERSE/01_RGB/01_color/only_logo.png",
)

st.image("/home/td/CI/CI_METAVERSE/01_RGB/01_color/CI_logo.png")

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            background-image: url(/home/td/CI/CI_METAVERSE/01_RGB/01_color/only_logo.png);
            background-repeat: no-repeat;
            width: 600px;
            height: 200px;
            padding-top: 20px;
            background-position: 20px 20px;
        }
        [data-testid="stSidebarNav"]::before {
            content: "Available demo list";
            margin-left: 20px;
            margin-top: 20px;
            font-size: 30px;
            position: relative;
            top: 100px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    # [Stable Diffusion 2.0](https://github.com/Stability-AI/stablediffusion) Demo
    ### Common Parameter Information
        - Prompt : The keyword of image what you want to generate
        - Seed : Seed of random initial value
            - If you want to generate different image using same prompt, you have to change seed number
        - Scale : Weights for the prompt
            - Increasing scale makes generated image follow more closely to the prompt.
    ### Available demo list
    - Text2Image
    """
)
st.image("/home/td/Project/stablediffusion/assets/stable-samples/txt2img/768/merged-0001.png")
st.markdown("""
    - Image2Image
    """)
st.image("/home/td/Project/stablediffusion/assets/stable-samples/depth2img/merged-0005.png")
st.markdown("""
    - Depth2Image
    """)
st.image("/home/td/Project/stablediffusion/assets/stable-samples/depth2img/merged-0000.png")
st.markdown("""
    - Inpainting
    """)
st.image("/home/td/Project/stablediffusion/assets/stable-inpainting/merged-leopards.png")
st.markdown("""
    - Superresolution
    """)
st.image("/home/td/Project/stablediffusion/assets/stable-samples/upscaling/merged-dog.png")
