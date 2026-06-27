import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

from torchvision import transforms
from PIL import Image

try:
    from model import build_model, build_binary_model
    from gradcam_explanation import generate_gradcam
    from lime_explaination import (
        create_predict_function,
        generate_lime
    )
except ModuleNotFoundError:
    from src.model import build_model, build_binary_model
    from src.gradcam_explanation import generate_gradcam
    from src.lime_explaination import (
        create_predict_function,
        generate_lime
    )

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="WasteVisionX: Explainable Classification and Anomaly Detection for Waste Sorting",
    layout="wide",
)

if "current_page" not in st.session_state:
    st.session_state.current_page = "main"

col1, col2 = st.columns([0.85, 0.15])

with col1:
    st.markdown(
        """
        <h1 style='font-size:30px; margin-top: -30px; margin-bottom: 5px;'>
            WasteVisionX: Explainable Classification and Anomaly Detection for Waste Sorting
        </h1>
        """,
        unsafe_allow_html=True
    )

with col2:
    if st.session_state.current_page == "main":
        if st.button("About", key="about_btn"):
            st.session_state.current_page = "about"
            st.rerun()
    else:
        if st.button("← Back", key="back_header_btn"):
            st.session_state.current_page = "main"
            st.rerun()

WASTE_CLASSES = [
    "Cardboard", "Food Organics", "Glass", "Metal",
    "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation"
]

GLASS_ANOMALY_CLASSES = ["Anomalous Glass (Broken)", "Normal Glass"]

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_waste(image_pil):
    """Predict waste category"""
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = main_model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs

def predict_glass_anomaly(image_pil):
    """Predict glass anomaly"""
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = glass_anomaly_model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs

def predict_glass_plastic(image_pil):
    """Predict plastic vs Glass"""
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = glass_plastic_model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs

@st.cache_resource
def load_main_model():

    model = build_model(9)

    model.load_state_dict(
        torch.load(
            "classification_saved_results/classification_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval() 
    return model

@st.cache_resource
def load_glass_anomaly_model():

    model = build_binary_model()

    model.load_state_dict(
        torch.load(
            "glass_anomaly_saved_results/glass_anomaly_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval() 
    return model

@st.cache_resource
def load_glass_plastic_model():


    model = build_binary_model()

    model.load_state_dict(
        torch.load(
            "glass_plastic_saved_results/glass_plastic_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()
    return model

main_model = load_main_model()
glass_anomaly_model = load_glass_anomaly_model()
glass_plastic_model = load_glass_plastic_model()

main_predict_fn = create_predict_function(
    main_model,
    transform,
    device
)

plastic_predict_fn = create_predict_function(
    glass_plastic_model,
    transform,
    device
)

if st.session_state.current_page == "main":

    st.markdown(
    """
    <p style="
        font-size:20px;
        font-weight:600;
        margin-bottom:8px;
        margin-top:-8px;">
        Upload a waste image for classification and analysis
    </p>
    """,
    unsafe_allow_html=True
)

    upload_col, preview_col = st.columns([2.5, 1.5], gap="large")

    with upload_col:

        upload_box = st.container(border=True)

        with upload_box:

            #st.markdown("### Upload an image")

            uploaded_file = st.file_uploader(
                "",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )

        

    with preview_col:

        

        preview_box = st.container(border=True)

        with preview_box:

            st.markdown(
                """
                <p style="
                    font-size:20px;
                    font-weight:600;
                    margin-bottom:8px;
                    margin-top:-8px;">
                    Image preview
                </p>
                """,
                unsafe_allow_html=True
            )

            if uploaded_file is not None:

                image_pil = Image.open(uploaded_file).convert("RGB")

                st.image(image_pil, width=150)
                #st.caption(uploaded_file.name)

            else:

                st.write("")
                #st.info("Image Preview")
                
    if uploaded_file is not None:
        p1_col, p2_col, p3_col = st.columns(3)

        probs = predict_waste(image_pil)
        predicted_class_idx = probs.argmax(dim=1).item()
        confidence = probs[0, predicted_class_idx].item() * 100

        final_predicted_class_idx = predicted_class_idx

        with p1_col:
            st.markdown("<p style='font-size: 20px; font-weight: bold; margin: -10px 0 1px 0;'>Waste Classification</p>", unsafe_allow_html=True)
            #st.markdown(f"<p style='font-size: 12px; color: #666; margin: 5px 0 10px 0;'>Predicted class: <span style='color: #0f8c3a; font-weight: bold;'>{WASTE_CLASSES[final_predicted_class_idx]}</span></p>", unsafe_allow_html=True)
            status_color = '#28a745'
            st.markdown(f"<div style='background-color: {status_color}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 9px; display: inline-block;'><b>Predicted Class: {WASTE_CLASSES[final_predicted_class_idx]}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 10px; margin: 2px 0 5px 0;'><b>Confidence:</b> {confidence:.2f}%</p>", unsafe_allow_html=True)


            prob_data = {
                WASTE_CLASSES[i]: probs[0, i].item() * 100
                for i in range(len(WASTE_CLASSES))
            }

            fig, ax = plt.subplots(figsize=(5, 2.5))

            classes = list(prob_data.keys())
            values = list(prob_data.values())

            y_pos = range(len(classes))
            bars = ax.barh(y_pos, values, height=0.6)

            for i, bar in enumerate(bars):
                if i == predicted_class_idx:
                    bar.set_color('#1f77b4')
                else:
                    bar.set_color('#d3d3d3')

            # Set labels and limits
            ax.set_yticks(y_pos)
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_xlabel('Probability (%)', fontsize=9)
            ax.set_xlim(0, 100)

            # Add percentage labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + 1, i, f'{value:.1f}%', va='center', fontsize=8)

            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with p2_col:

            if predicted_class_idx == 2:  # Glass
                st.markdown(
                    "<p style='font-size:20px; font-weight:bold; margin:-10px 0 1px 0;'>Glass vs Plastic - Verification</p>",
                    unsafe_allow_html=True
                )

                # Use glass_plastic_model (0=Glass, 1=Plastic)
                gp_probs = predict_glass_plastic(image_pil)
                gp_idx = gp_probs.argmax(dim=1).item()
                gp_confidence = gp_probs[0, gp_idx].item() * 100

                if gp_idx == 0:  # Glass confirmed
                    final_predicted_class_idx = 2
                    status_color = "#28a745"

                    st.markdown(
                        f"<div style='background-color:{status_color}; color:white; padding:3px 8px; border-radius:4px; font-size:9px; display:inline-block;'><b>Predicted Class: Glass</b></div>",
                        unsafe_allow_html=True
                    )
                    #st.markdown("<div style='background-color: #d4edda; padding: 4px 8px; border-radius: 3px; border-left: 3px solid #28a745;'><p style='font-size: 9px; margin: 0; color: #155724;'><b>Predicted Class: Glass</b></p></div>", unsafe_allow_html=True)
                else:  # Actually Plastic
                    final_predicted_class_idx = 6
                    
                    st.markdown("<div style='background-color: #fff3cd; padding: 4px 8px; border-radius: 3px; border-left: 3px solid #ffc107;'><p style='font-size: 9px; margin: 0; color: #856404;'><b>Might be: Plastic</b></p></div>", unsafe_allow_html=True)

                class_name = "Glass" if gp_idx == 0 else "Plastic"
                #st.markdown(f"<p style='font-size: 12px; font-weight: bold; margin: 6px 0 0 0;'>{class_name} <span style='font-size: 10px; color: #666; font-weight: normal;'>Confidence: {gp_confidence:.2f}%</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 10px; margin: 2px 0 5px 0;'><b>Confidence:</b> {gp_confidence:.2f}%</p>", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(3.5, 1.8))
                bars = ax.barh(
                    ["Glass", "Plastic"],
                    [gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]
                )
                bars[gp_idx].set_color('#28a745' if gp_idx == 0 else '#dc3545')
                bars[1 - gp_idx].set_color('#d3d3d3')
                ax.set_xlabel('Probability (%)', fontsize=9)
                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlim(0, 100)
                for i, v in enumerate([gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]):
                    ax.text(v + 1, i, f'{v:.1f}%', va='center')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            elif predicted_class_idx == 6:  # Plastic
                st.markdown(
                    "<p style='font-size:20px; font-weight:bold; margin:-10px 0 1px 0;'>Glass vs Plastic - Verification</p>",
                    unsafe_allow_html=True
                )
                # Use glass_plastic_model (0=Glass, 1=Plastic)
                gp_probs = predict_glass_plastic(image_pil)
                gp_idx = gp_probs.argmax(dim=1).item()
                gp_confidence = gp_probs[0, gp_idx].item() * 100

                if gp_idx == 0:  # Verification says Glass
                    final_predicted_class_idx = 2  # Glass index
                    
                    st.markdown("<div style='background-color: #fff3cd; padding: 4px 8px; border-radius: 3px; border-left: 3px solid #ffc107;'><p style='font-size: 9px; margin: 0; color: #856404;'><b> Might be: Glass</b></p></div>", unsafe_allow_html=True)
                else:  # Plastic confirmed
                    final_predicted_class_idx = 6
                    status_color = "#28a745"
                    st.markdown(
                        f"<div style='background-color:{status_color}; color:white; padding:3px 8px; border-radius:4px; font-size:9px; display:inline-block;'><b>Predicted Class: Plastic</b></div>",
                        unsafe_allow_html=True
                    )

                class_name = "Glass" if gp_idx == 0 else "Plastic"
                st.markdown(f"<p style='font-size: 10px; margin: 2px 0 5px 0;'><b>Confidence:</b> {gp_confidence:.2f}%</p>", unsafe_allow_html=True)


                fig, ax = plt.subplots(figsize=(3.5, 1.8))
                bars = ax.barh(
                    ["Glass", "Plastic"],
                    [gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]
                )
                bars[gp_idx].set_color(
                    '#28a745' if gp_idx == 0 else '#dc3545'
                )
                bars[1 - gp_idx].set_color('#d3d3d3')
                ax.set_xlabel('Probability (%)', fontsize=9)
                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlim(0, 100)
                for i, v in enumerate([gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]):
                    ax.text(v + 1, i, f'{v:.1f}%', va='center')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            else:
                st.markdown(
                    "<p style='font-size:20px; font-weight:bold; margin:-10px 0 1px 0;'>Glass vs Plastic - Verification</p>",
                    unsafe_allow_html=True
                )
                st.markdown(f"<div style='background-color: #28a745; color: white; padding: 3px 8px; border-radius: 4px; font-size: 9px; display: inline-block;'><b>Only for Glass & Plastic</b></div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(3.5, 1.8))
                bars = ax.barh(
                    ["Glass", "Plastic"],
                    [0, 0]
                )
                ax.set_xlabel('Probability (%)', fontsize=9)
                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlim(0, 100)
                fig.tight_layout()
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with p3_col:
            if final_predicted_class_idx == 2:  # Glass
                st.markdown("<p style='font-size: 20px; font-weight: bold; margin: -10px 0 1px 0;'>Glass Anomaly Detection</p>", unsafe_allow_html=True)

                anomaly_probs = predict_glass_anomaly(image_pil)
                anomaly_idx = anomaly_probs.argmax(dim=1).item()
                anomaly_confidence = anomaly_probs[0, anomaly_idx].item() * 100

                status_text = GLASS_ANOMALY_CLASSES[anomaly_idx]
                status_color = '#28a745' if anomaly_idx == 1 else '#dc3545'
                st.markdown(f"<div style='background-color: {status_color}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 9px; display: inline-block;'><b>Predicted Class: {status_text}</b></div>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size: 10px; margin: 2px 0 5px 0;'><b>Confidence:</b> {anomaly_confidence:.2f}%</p>", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(3.5, 1.8))
                bars = ax.barh(
                    GLASS_ANOMALY_CLASSES,
                    [anomaly_probs[0, 0].item() * 100, anomaly_probs[0, 1].item() * 100]
                )
                bars[anomaly_idx].set_color('#28a745' if anomaly_idx == 1 else '#dc3545')
                bars[1 - anomaly_idx].set_color('#d3d3d3')
                ax.set_xlabel('Probability (%)', fontsize=9)
                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlim(0, 100)
                for i, v in enumerate([anomaly_probs[0, 0].item() * 100, anomaly_probs[0, 1].item() * 100]):
                    ax.text(v + 1, i, f'{v:.1f}%', va='center')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            else:
                st.markdown("<p style='font-size: 20px; font-weight: bold; margin: -10px 0 1px 0;'>Glass Anomaly Detection</p>", unsafe_allow_html=True)
        
                st.markdown(f"<div style='background-color: #28a745; color: white; padding: 3px 8px; border-radius: 4px; font-size: 9px; display: inline-block;'><b>Anomaly detection available only for Glass</b></div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(3.5, 1.8))
                bars = ax.barh(
                    [ "Anomalous Glass (Broken)", "Normal Glass"],
                    [0, 0]
                )
                ax.set_xlabel('Probability (%)', fontsize=9)
                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlim(0, 100)
                fig.tight_layout()
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-top: -10px;'>Explainable AI (XAI Validation)</p>", unsafe_allow_html=True)
        #st.markdown("<p style='font-size: 11px;'>Validating the classified class detection with Grad-CAM and LIME</p>", unsafe_allow_html=True)

        if final_predicted_class_idx == 6:
            explain_model = glass_plastic_model
            predict_fn = plastic_predict_fn
        else:
            explain_model = main_model
            predict_fn = main_predict_fn
           

        xai_col1, xai_col2 = st.columns(2)

        with xai_col1:
            st.markdown("<p style='font-size: 15px; font-weight: bold;'>Grad-CAM Visualization</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 12px;'>Shows which image regions the model focused on (Red = High importance, Blue = Low importance)</p>", unsafe_allow_html=True)

            with st.spinner("Generating Grad-CAM..."):
                try:
                    gradcam_img = generate_gradcam(
                        explain_model,
                        image_pil,
                        transform,
                        device
                    )                    
                    st.image(gradcam_img, caption="Grad-CAM", width=400)
                except Exception as e:
                    st.error(f"Grad-CAM Error: {str(e)}")

        with xai_col2:
            st.markdown("<p style='font-size: 15px; font-weight: bold;'>LIME Explanation</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 12px;'>Shows important feature regions.</p>", unsafe_allow_html=True)

            with st.spinner("Generating LIME... (this may take a moment)"):
                try:

                    lime_img = generate_lime(
                        image_pil,
                        predict_fn
                    )

                    st.image(
                        lime_img,
                        caption="LIME",
                        width=400
                        #use_column_width=True
                    )
                except Exception as e:
                    st.error(f"LIME Error: {str(e)}")

        #st.markdown(f"<p style='font-size: 10px; margin-top: 5px;'>Confidence: {lime_confidence:.2f}%</p>", unsafe_allow_html=True)


        st.markdown("<p style='font-size: 16px; font-weight: bold; margin-top: 15px;'>Final Prediction: {}</p>".format(WASTE_CLASSES[final_predicted_class_idx]), unsafe_allow_html=True)

if st.session_state.current_page == "about":
    if st.button("← Back to Main", key="back_btn"):
        st.session_state.current_page = "main"
        st.rerun()

    st.divider()
    st.header("About This Project")

    st.subheader("Project Goal")
    st.write(
        "The goal of this project is to develop a comprehensive machine learning solution for waste sorting. "
        "It includes waste classification, anomaly detection for glass, "
        "and explainable AI techniques (Grad-CAM and LIME) to validate model predictions.")

    st.subheader("Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**Classification of waste into 9 categories:**\n\n"
            "-> Cardboard\n\n"
            "-> Food Organics\n\n"
            "-> Glass\n\n"
            "-> Metal\n\n"
            "-> Miscellaneous Trash\n\n"
            "-> Paper\n\n"
            " -> Plastic\n\n"
            " -> Textile Trash\n\n"
            " -> Vegetation"
        )

    with col2:
        st.info(
            "**Plastic vs Glass**\n\n"
            "-> Identifies if the classification is Plastic or Glass:\n\n"
            "-> Glass and Plastic are often confused, so this step ensures accurate classification."
        )

    with col3:
        st.info(
            "**Glass Anomaly Detection**\n\n"
            "Detects defective waste:\n\n"
            "-> Glass: Normal vs Broken\n\n"
            "-> Identifies quality issues"
        )

    st.subheader("Explainability")
    st.write(
        "Understand model decisions:\n\n"
        "- **Grad-CAM heatmaps** - Visual attention maps\n"
        "- **LIME explanations** - Local interpretable model explanations"
    )

    st.subheader("Technical Stack")
    st.write(
        "- **Model:** ResNet50 (Transfer Learning)\n"
        "- **Framework:** PyTorch\n"
        "- **Frontend:** Streamlit\n"
        "- **XAI:** Grad-CAM + LIME\n"
        "- **Accuracy:** 80% on waste classification"
    )