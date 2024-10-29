import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import uuid
import matplotlib.pyplot as plt
from PIL import Image, ImageCms
from io import BytesIO

from streamlit_image_select import image_select

# Import Firebase libraries
import json
from google.oauth2 import service_account
from google.cloud import firestore

# Set page configuration
st.set_page_config(page_title="Nature's Palette", layout="wide")

# Hide Streamlit's default menu and footer (optional)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Header
st.title("GCIN4001 Capstone Research Project: A Study of Color and Preference")

# Consent Form
def show_consent_form():
    st.header("Informed Consent Form for Participants")
    consent_text = """
    **Study Title:** GCIN4001 Capstone Research Project: A Study of Color and Preference

    **Principal Investigator:** Mr. Yau, u3584220@connect.hku.hk, School of Modern Languages and Cultures

    **Purpose of the Study:**
    The purpose of this study is to investigate how different color formats (CMYK, eciRGB v2, RGB) impact the perception of natural objects. Your participation will contribute to a better understanding of color perception and its applications in design and visual arts.

    **Procedures:**
    You will complete 15 trials, consisting of FIVE sets of images presented THREE times each. Each set contains the same image in three different color formats. Your task is to select the image you find most visually appealing in each trial. The order of images within each set will be randomized for each of the three presentations.

    **Confidentiality and Data Usage:**
    Your responses will be recorded anonymously. No personally identifiable information will be collected. The data gathered will be used solely for research purposes and may be published in academic journals or conferences in an aggregated form.

    **Voluntary Participation:**
    Your participation is voluntary. You may withdraw from the study at any time by closing the browser window without any penalty.

    **Agreement:**
    By checking the box below, you acknowledge that you have read and understood the information provided above and agree to participate in this study.
    """
    st.markdown(consent_text)
    consent_given = st.checkbox("I have read the above information and consent to participate in this study.")

    return consent_given

# Initialize session state variables
if 'consent_given' not in st.session_state:
    st.session_state.consent_given = False

if 'responses' not in st.session_state:
    st.session_state.responses = []

if 'current_task_index' not in st.session_state:
    st.session_state.current_task_index = 0

if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())  # Unique identifier for each participant

# Load Firebase credentials from Streamlit secrets
key_json = st.secrets["textkey"]
key_dict = json.loads(key_json)
credentials = service_account.Credentials.from_service_account_info(key_dict)
db = firestore.Client(credentials=credentials, project=key_dict["project_id"])

# Function to save responses to Firestore
def save_responses_to_firestore(responses_list):
    try:
        user_id = responses_list[0]['user_id']
        user_collection = db.collection(user_id)
        
        # Organize responses by object
        object_responses = {}
        for record in responses_list:
            obj = record['object']
            attempt = record['repeat']
            selected_color_space = record['selected_color_space']
            
            if obj not in object_responses:
                object_responses[obj] = {}
            object_responses[obj][f'attempt_{attempt}'] = selected_color_space
        
        # Save each object's responses
        for obj, attempts in object_responses.items():
            doc_ref = user_collection.document(obj)
            doc_ref.set(attempts)
    except Exception as e:
        st.error(f"Error saving to Firestore: {e}")

# Load images
def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

# Function to convert image colors
def convert_image_colors(image, color_space):
    if color_space == 'CMYK':
        # Convert image from sRGB to CMYK using ICC profiles
        try:
            # Paths to ICC profiles
            srgb_profile = 'icc_profiles/sRGB_v4_ICC_preference.icc'  # sRGB profile
            cmyk_profile = 'icc_profiles/APTEC_PC10_CardBoard_2023_v1.icc'  # CMYK profile

            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert from sRGB to CMYK
            img_cmyk = ImageCms.profileToProfile(image, srgb_profile, cmyk_profile, outputMode='CMYK')
            # Convert back to RGB for display
            img_rgb = ImageCms.profileToProfile(img_cmyk, cmyk_profile, srgb_profile, outputMode='RGB')
            return img_rgb
        except Exception as e:
            st.error(f"Error converting to CMYK: {e}")
            return image
    elif color_space == 'eciRGB v2':
        # Convert image from sRGB to eciRGB v2
        try:
            # Paths to ICC profiles
            srgb_profile = 'icc_profiles/sRGB_v4_ICC_preference.icc'
            ecirgb_profile = 'icc_profiles/eciRGB_v2.icc'

            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert from sRGB to eciRGB v2
            img_ecirgb = ImageCms.profileToProfile(image, srgb_profile, ecirgb_profile, outputMode='RGB')
            return img_ecirgb
        except Exception as e:
            st.error(f"Error converting to eciRGB v2: {e}")
            return image
    elif color_space == 'RGB':
        # No conversion needed
        return image
    else:
        return image

# Consent Step
if not st.session_state.consent_given:
    consent = show_consent_form()
    if consent:
        st.session_state.consent_given = True
        st.rerun()  # Reload the app to proceed
    else:
        st.stop()  # Stop the app until consent is given

# Hide consent form after giving consent
if st.session_state.consent_given:
    # Introduction text
    st.write("""
    You're ready to begin!
    
    In this study, you will be presented with sets of images. Each set features variations in color formatting. Your task is to select the image from each set that you find most visually appealing. The images within each set will be presented in a random order.

    Your responses will remain completely anonymous and will be used solely for research purposes.

    Let's begin!
    """)

    # List of objects
    object_images = [f for f in os.listdir('data/objects') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Ensure images are found
    if not object_images:
        st.error("No images found in the 'data/objects/' directory.")
        st.stop()

    # Ensure consistent order for all participants
    object_images.sort()

    # Create task list: each object repeated 3 times
    task_list = []
    for obj_image in object_images:
        for repeat in range(1, 4):  # Repeats 1 to 3
            task_list.append({'object': obj_image, 'repeat': repeat})

    # Total number of tasks
    total_tasks = len(task_list)

    # Progress indicator
    progress = (st.session_state.current_task_index) / total_tasks
    st.progress(progress)

    # Main content
    if st.session_state.current_task_index < total_tasks:
        task = task_list[st.session_state.current_task_index]
        obj_image = task['object']
        repeat = task['repeat']

        st.header(f"Task {st.session_state.current_task_index + 1} of {total_tasks}")
        st.subheader(f"Object: {os.path.splitext(obj_image)[0].capitalize()} (Repeat {repeat}/3)")

        # Load original image
        image = load_image(f'data/objects/{obj_image}')
        if image is None:
            st.error(f"Could not load image {obj_image}. Please contact the administrator.")
            st.stop()

        # Generate different color versions
        color_spaces = ['CMYK', 'eciRGB v2', 'RGB']

        # Create variations with different color formats
        images_with_formats = []
        for cs in color_spaces:
            img_converted = convert_image_colors(image, cs)
            images_with_formats.append((cs, img_converted))

        # Shuffle the images differently for each repeat
        rng = random.Random(str(st.session_state.user_id) + str(st.session_state.current_task_index))
        rng.shuffle(images_with_formats)

        # Prepare images and captions for selection
        images = [img for _, img in images_with_formats]
        captions = [f"Option {i+1}" for i in range(len(images))]
        color_spaces_shuffled = [cs for cs, _ in images_with_formats]

        # Allow participant to select image by clicking on it
        st.write("Please select the image you like the most:")
        selected_idx = image_select(
            label="",
            images=images,
            captions=captions,
            return_value='index',
            use_container_width=True,
            key=f"selection_{st.session_state.current_task_index}"
        )

        # Provide option to view each image separately without revealing color format
        with st.expander("View Images in Full Size"):
            for idx, img in enumerate(images):
                st.subheader(f"Option {idx+1}")
                st.image(img, use_column_width=True)

        # Ensure a selection is made
        if selected_idx is None:
            st.warning("Please make a selection before proceeding.")
            st.stop()

        # Record the response
        st.session_state.responses.append({
            'user_id': st.session_state.user_id,
            'task_index': st.session_state.current_task_index + 1,
            'object': obj_image,
            'repeat': repeat,
            'selected_option': selected_idx + 1,  # Option number
            'selected_color_space': color_spaces_shuffled[selected_idx],
            'option_1_color_space': color_spaces_shuffled[0],
            'option_2_color_space': color_spaces_shuffled[1],
            'option_3_color_space': color_spaces_shuffled[2],
        })

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.current_task_index > 0:
                if st.button("Previous"):
                    st.session_state.current_task_index -= 1
                    # Remove the last response
                    if st.session_state.responses:
                        st.session_state.responses.pop()
                    st.rerun()
        with col2:
            if st.session_state.current_task_index < total_tasks - 1:
                if st.button("Next"):
                    st.session_state.current_task_index += 1
                    st.rerun()
            else:
                if st.button("Submit"):
                    # Save responses to Firestore
                    save_responses_to_firestore(st.session_state.responses)

                    st.success("You have completed the study. Thank you for your participation!")
                    st.balloons()

                    # Prepare responses DataFrame
                    responses_df = pd.DataFrame(st.session_state.responses)

                    # Clear session state after submission
                    st.session_state.clear()

                    # Display participant's own data
                    st.header("Your Selections and Preferences")

                    st.subheader("Your Selections:")
                    st.dataframe(responses_df[['task_index', 'object', 'repeat', 'selected_color_space']], use_container_width=True)

                    st.subheader("Summary of Your Preferences:")
                    preference_counts = responses_df['selected_color_space'].value_counts()
                    fig, ax = plt.subplots(figsize=(10,6))
                    preference_counts.plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_xlabel('Color Space')
                    ax.set_ylabel('Number of Selections')
                    ax.set_title('Your Color Space Preferences')
                    st.pyplot(fig)

                    # Save the chart to a buffer
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)

                    # Provide download button for the chart image
                    st.download_button(
                        label="Download Chart as PNG",
                        data=buf,
                        file_name='your_preferences_chart.png',
                        mime='image/png',
                    )

                    # Provide download option for participant's own data
                    csv_data = responses_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Your Responses as CSV",
                        data=csv_data,
                        file_name='your_responses.csv',
                        mime='text/csv',
                    )

                    st.stop()
    else:
        st.success("You have completed the study. Thank you for your participation!")
        st.balloons()

        # Prepare responses DataFrame
        responses_df = pd.DataFrame.from_dict(st.session_state.responses, orient='index')

        # Clear session state after submission
        st.session_state.clear()

        # Display participant's own data
        st.header("Your Selections and Preferences")

        st.subheader("Your Selections:")
        st.dataframe(responses_df[['task_index', 'object', 'repeat', 'selected_color_space']], use_container_width=True)

        st.subheader("Summary of Your Preferences:")
        preference_counts = responses_df['selected_color_space'].value_counts()
        fig, ax = plt.subplots(figsize=(10,6))
        preference_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_xlabel('Color Space')
        ax.set_ylabel('Number of Selections')
        ax.set_title('Your Color Space Preferences')
        st.pyplot(fig)

        # Save the chart to a buffer
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Provide download button for the chart image
        st.download_button(
            label="Download Chart as PNG",
            data=buf,
            file_name='your_preferences_chart.png',
            mime='image/png',
        )

        # Provide download option for participant's own data
        csv_data = responses_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Your Responses as CSV",
            data=csv_data,
            file_name='your_responses.csv',
            mime='text/csv',
        )

        st.stop()
