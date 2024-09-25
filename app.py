import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import uuid
import matplotlib.pyplot as plt
from PIL import Image

from streamlit_image_select import image_select

# Import Firebase libraries
import json
from google.oauth2 import service_account
from google.cloud import firestore

# For downloading chart image
from io import BytesIO

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

# Add CSS to fix image scaling
st.markdown(
    """
    <style>
    img[data-baseweb="image"] {
        object-fit: contain !important;
        width: 100% !important;
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("Nature's Palette: Exploring Color Perception")

st.write("""
Welcome to the Nature's Palette study!

In this study, you will see different visual representations of natural objects. Each object is displayed three times with different color formats shuffled randomly. Your task is to select the image you like the most each time.

Please note that your participation is anonymous, and your responses will be used solely for research purposes.

Let's begin!
""")

# Initialize session state variables
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
        # Convert image to CMYK and back to RGB
        cmyk_image = image.convert('CMYK')
        return cmyk_image.convert('RGB')
    elif color_space == 'Pantone':
        # Simulate Pantone conversion
        image = image.convert('P', palette=Image.ADAPTIVE, colors=20)
        return image.convert('RGB')
    elif color_space == 'RGB':
        # No conversion needed
        return image
    else:
        return image

# Function to record response
def record_response(selected_idx):
    # Ensure that st.session_state.responses is a list
    if 'responses' not in st.session_state or not isinstance(st.session_state.responses, list):
        st.session_state.responses = []

    # Remove existing response for this task index if any (to prevent duplicates)
    response_indices = [i for i, resp in enumerate(st.session_state.responses) if resp['task_index'] == st.session_state.current_task_index + 1]
    for idx in sorted(response_indices, reverse=True):
        del st.session_state.responses[idx]

    # Add the new response
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
    color_spaces = ['CMYK', 'Pantone', 'RGB']

    # Create variations with different color formats
    images_with_formats = []
    for cs in color_spaces:
        img_converted = convert_image_colors(image, cs)
        images_with_formats.append((cs, img_converted))

    # Shuffle the images differently for each repeat
    rng = random.Random(str(st.session_state.user_id) + str(st.session_state.current_task_index))
    rng.shuffle(images_with_formats)

    # Prepare images and captions for selection
    images = [img.resize((400, 400)) for _, img in images_with_formats]  # Resize images for consistency
    captions = [f"Option {i+1}" for i in range(len(images))]
    color_spaces_shuffled = [cs for cs, _ in images_with_formats]

    # Allow participant to select image by clicking on it
    with st.container():
        st.write("Please select the image you like the most:")
        try:
            selected_idx = image_select(
                label="",
                images=images,
                captions=captions,
                use_container_width=True,
                return_value='index',
                key=f"selection_{st.session_state.current_task_index}"
            )
        except Exception as e:
            st.error(f"Error with image selection: {e}")
            st.stop()

    # Ensure a selection is made
    if selected_idx is None:
        st.warning("Please make a selection before proceeding.")
        st.stop()

    # Record the response
    record_response(selected_idx)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.session_state.current_task_index > 0:
            if st.button("Previous"):
                st.session_state.current_task_index -= 1
                # Remove the last response
                if st.session_state.responses:
                    st.session_state.responses.pop()
                st.rerun()
    with col2:
        pass  # Empty column for spacing
    with col3:
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
                with st.container():
                    st.header("Your Selections and Preferences")

                    st.subheader("Your Selections:")
                    st.dataframe(responses_df[['task_index', 'object', 'repeat', 'selected_color_space']])

                    st.subheader("Summary of Your Preferences:")
                    preference_counts = responses_df['selected_color_space'].value_counts()
                    fig, ax = plt.subplots(figsize=(8,6))
                    preference_counts.plot(kind='bar', ax=ax)
                    ax.set_xlabel('Color Space')
                    ax.set_ylabel('Number of Selections')
                    ax.set_title('Your Color Space Preferences')

                    # Display the chart
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
    responses_df = pd.DataFrame(st.session_state.responses)

    # Clear session state after submission
    st.session_state.clear()

    # Display participant's own data
    with st.container():
        st.header("Your Selections and Preferences")

        st.subheader("Your Selections:")
        st.dataframe(responses_df[['task_index', 'object', 'repeat', 'selected_color_space']])

        st.subheader("Summary of Your Preferences:")
        preference_counts = responses_df['selected_color_space'].value_counts()
        fig, ax = plt.subplots(figsize=(8,6))
        preference_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Color Space')
        ax.set_ylabel('Number of Selections')
        ax.set_title('Your Color Space Preferences')

        # Display the chart
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