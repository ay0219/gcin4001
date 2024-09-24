import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from colormath.color_objects import sRGBColor, CMYKColor, LabColor
from colormath.color_conversions import convert_color
import os
import random
import uuid
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Nature's Palette", layout="wide")

# Hide Streamlit style (optional)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Header
st.title("Nature's Palette: Exploring Color Perception")

st.write("""
Welcome to the Nature's Palette study!

In this study, you will see different visual representations of natural objects. Each object is displayed in three different color renditions. Your task is to select the one that you perceive as:

- **Most Accurate** in representing the natural object's colors.
- **Most Visually Appealing** in terms of color harmony and aesthetics.

Let's begin!
""")

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = []

if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())  # Unique identifier for each participant

# Load images
def load_image(image_path):
    return Image.open(image_path)

# Function to convert image colors
def convert_image_colors(image, color_space):
    if color_space == 'CMYK':
        # Convert image to CMYK
        cmyk_image = image.convert('CMYK')
        return cmyk_image.convert('RGB')  # Convert back to RGB for display
    elif color_space == 'Pantone':
        # Simulate Pantone conversion (simplified)
        # A real Pantone conversion would require a licensed library
        # Here, we'll adjust the colors to mimic a limited palette
        image = image.convert('P', palette=Image.ADAPTIVE, colors=50)
        return image.convert('RGB')
    elif color_space == 'RGB':
        # No conversion needed
        return image
    else:
        return image

# List of objects
object_images = [f for f in os.listdir('data/objects') if f.endswith(('.jpg', '.png'))]

# Main loop through objects
for obj_image in object_images:
    st.header(f"Object: {os.path.splitext(obj_image)[0].capitalize()}")
    col1, col2, col3 = st.columns(3)
    
    # Load original image
    image = load_image(f'data/objects/{obj_image}')
    
    # Generate different color versions
    color_spaces = ['CMYK', 'Pantone', 'RGB']
    image_variations = []
    for cs in color_spaces:
        img_converted = convert_image_colors(image, cs)
        image_variations.append((cs, img_converted))
    
    # Shuffle images
    random.shuffle(image_variations)
    
    # Display images without revealing color space
    with col1:
        st.image(image_variations[0][1], use_column_width=True)
        selection1 = st.radio("Select for this image:", ["Most Accurate", "Most Appealing", "Neither"], key=f"{obj_image}_1")
    with col2:
        st.image(image_variations[1][1], use_column_width=True)
        selection2 = st.radio("Select for this image:", ["Most Accurate", "Most Appealing", "Neither"], key=f"{obj_image}_2")
    with col3:
        st.image(image_variations[2][1], use_column_width=True)
        selection3 = st.radio("Select for this image:", ["Most Accurate", "Most Appealing", "Neither"], key=f"{obj_image}_3")
    
    # Store responses
    st.session_state.responses.append({
        'user_id': st.session_state.user_id,
        'object': obj_image,
        'option_1_color_space': image_variations[0][0],
        'option_1_selection': selection1,
        'option_2_color_space': image_variations[1][0],
        'option_2_selection': selection2,
        'option_3_color_space': image_variations[2][0],
        'option_3_selection': selection3,
    })

# Submit button
if st.button("Submit Responses"):
    # Save responses to CSV
    responses_df = pd.DataFrame(st.session_state.responses)
    responses_file = 'data/responses/responses.csv'
    if os.path.exists(responses_file):
        # Append to existing CSV
        existing_df = pd.read_csv(responses_file)
        combined_df = pd.concat([existing_df, responses_df], ignore_index=True)
        combined_df.to_csv(responses_file, index=False)
    else:
        # Create new CSV
        responses_df.to_csv(responses_file, index=False)
    
    st.success("Your responses have been recorded. Thank you!")
    st.balloons()
    
    # Data analysis and visualization
    st.header("Preliminary Results")
    
    # Load all responses
    df = pd.read_csv(responses_file)
    
    # Chart: Selection counts per color space
    selection_counts = df.melt(id_vars=['user_id', 'object'], value_vars=['option_1_selection', 'option_2_selection', 'option_3_selection'], var_name='option', value_name='selection')
    color_spaces = df.melt(id_vars=['user_id', 'object'], value_vars=['option_1_color_space', 'option_2_color_space', 'option_3_color_space'], var_name='option', value_name='color_space')
    merged = pd.merge(selection_counts, color_spaces, on=['user_id', 'object', 'option'])
    
    # Filter out 'Neither' selections
    filtered = merged[merged['selection'] != 'Neither']
    
    # Group by color space and selection
    group_counts = filtered.groupby(['color_space', 'selection']).size().unstack(fill_value=0)
    
    # Plotting
    fig, ax = plt.subplots()
    group_counts.plot(kind='bar', stacked=True, ax=ax)
    plt.title('Selections per Color Space')
    plt.xlabel('Color Space')
    plt.ylabel('Count')
    st.pyplot(fig)
    
    st.write("You can download the full responses data below:")
    st.download_button(
        label="Download Responses as CSV",
        data=combined_df.to_csv(index=False).encode('utf-8'),
        file_name='responses.csv',
        mime='text/csv',
    )