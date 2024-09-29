import time
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import io

def main():
    # Set the page configuration
    st.set_page_config(page_title="Drawing Pad", layout="wide")

    # Initialize session state to track if the start button has been pressed
    if "start_pressed" not in st.session_state:
        st.session_state.start_pressed = False

    # Center the logo and the start button using Streamlit columns
    col1, col2, col3 = st.columns([1, 2, 1])

    if not st.session_state.start_pressed:
        with col2:
            # Display the logo
            logo = Image.open("../views/logo.png")
            st.image(logo, width=400)

            # Slider for timer (15-90)
            timer = st.slider("Set Timer (in seconds):", min_value=15, max_value=90, value=30)

            # Slider for number of players (1-8)
            num_players = st.slider("Set Number of Players:", min_value=1, max_value=8, value=4)

            # Start button to show the drawing pad
            if st.button("Start", key="start_button"):
                # Mark start as pressed
                st.session_state.start_pressed = True

                # Store the current value of the timer slider in session state
                st.session_state.timer_duration = timer

                # Store the current value of the number of players slider in session state
                st.session_state.num_players = num_players

                st.current_turn = 0
                

    # If the start button has been pressed, show the drawing pad and related options
    if st.session_state.start_pressed:
        # Create a drawing canvas with an eraser option
        drawing_mode = st.radio("Drawing Mode:", options=["Draw", "Eraser"])

        # Options for brush sizes
        brush_size = st.select_slider("Select Brush Size:", options=list(range(1, 21)), value=2)

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

                # Display the submitted image at the bottom
                # st.subheader("Your Submitted Drawing:")
                # st.image(image, caption="Drawing", use_column_width=True)
                

                # Reset the timer to 30 seconds
                st.session_state.timer_duration = st.session_state.timer_duration

                # Increment the current turn
                st.current_turn = st.current_turn + 1

                # Check if the current turn equals the number of players
                if st.current_turn == st.session_state.num_players:
                    st.success("All players have submitted their drawings!")
                    # Perform any additional actions here, such as resetting the game or displaying results
                    st.session_state.start_pressed = False  # Reset the start button state
                    st.current_turn = 0  # Reset the turn counter

            else:
                st.warning("Please draw something before submitting.")
        
        # Input slider for the countdown duration
        countdown_duration = st.slider("Set Countdown Timer (seconds):", min_value=1, max_value=90, value= st.session_state.timer_duration)

        countdown_placeholder = st.empty()

        for i in range(countdown_duration, 0, -1):
            # Update the countdown number
            countdown_placeholder.markdown(f"## Time Remaining: {i} seconds")
            time.sleep(1)

        # Increment the current turn
            st.current_turn = st.current_turn + 1
            st.session_state.timer_duration = st.session_state.timer_duration
            

if __name__ == '__main__':
    main()
