import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import io

def main():
    # Set the page configuration
    st.set_page_config(page_title="Drawing Pad", layout="wide")

    # Title of the app
    st.title("Drawing Pad")


    # Create a drawing canvas with an eraser option
    drawing_mode = st.radio("Drawing Mode:", options=["Draw", "Eraser"])

    # Options for brush sizes
    brush_size = st.select_slider("Select Brush Size:", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], value=2)

    # Set the background color and the color used when drawing
    background_color = "white"
    stroke_color = "black" if drawing_mode == "Draw" else background_color

    # Create the drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # Transparent fill color
        stroke_width=brush_size,
        stroke_color=stroke_color,
        background_color=background_color,
        width=700,
        height=500,
        drawing_mode="freedraw",  # Always use freedraw mode
        key="canvas"
    )

    # Variable to store the submitted image for display
    submitted_image = None

    # Button to submit the drawing
    if st.button("Submit Drawing"):
        if canvas_result.image_data is not None:
            # Convert the drawn image to a PIL Image
            image = Image.fromarray(canvas_result.image_data.astype(np.uint8))

            # Convert to RGB if the image has an alpha channel
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            # Save the image to a BytesIO object as JPEG
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")  # Save as JPEG
            buffer.seek(0)

            # Send the image to the server (example endpoint)
            # response = requests.post(
            #     "http://your-server-endpoint/upload",  # Replace with your server endpoint
            #     files={"drawing.jpg": buffer}  # Use .jpg extension
            # )

            # For now, we will skip the server call for display purposes
            # Handle the server response
            # if response.status_code == 200:
            #     st.success("Drawing submitted successfully!")
            # else:
            #     st.error("Failed to submit drawing.")

            # Store the submitted image for display
            submitted_image = image  # Store the image for later display

        else:
            st.warning("Please draw something before submitting.")

    # Display the submitted image at the bottom if it exists
    if submitted_image is not None:
        st.subheader("Your Submitted Drawing:")
        st.image(submitted_image, caption="Drawing", use_column_width=True)

if __name__ == '__main__':
    main()