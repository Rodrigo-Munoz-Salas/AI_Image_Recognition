import time
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

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
            logo = Image.open("./views/logo.png")
            st.image(logo, width=400)



            # Slider for number of players (1-8)
            num_players = st.slider("Set Number of Players:", min_value=1, max_value=8, value=4)

            # Start button to show the drawing pad
            if st.button("Start", key="start_button"):
                # Mark start as pressed
                st.session_state.start_pressed = True



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

        # Initialize submitted_image as None to avoid UnboundLocalError
        submitted_image = None

        # Button to submit the drawing
        if st.button("Submit Drawing"):
            if canvas_result.image_data is not None:
                # Convert the drawn image to a PIL Image
                image = Image.fromarray(canvas_result.image_data.astype(np.uint8))

                # Convert to RGB if the image has an alpha channel
                if image.mode == 'RGBA':
                    image = image.convert('RGB')

                # Increment the current turn
                st.current_turn = st.current_turn + 1

                # Check if the current turn equals the number of players
                if st.current_turn >= st.session_state.num_players:
                    st.success("All players have submitted their drawings!")

                    # Perform any additional actions here, such as resetting the game or displaying results
                    st.session_state.start_pressed = False  # Reset the start button state
                    st.current_turn = 0  # Reset the turn counter



                submitted_image = image  # Store the image for later display

            else:
                st.warning("Please draw something before submitting.")
        

            # Increment the current turn
            st.current_turn = st.current_turn + 1
            

        # =========================================================== #

        # Transforming the images
        transform = transforms.Compose([
            transforms.ToTensor(), # Converting pillow images to PyToech tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # (channels x height x width) Normalizing images with a mean of 0.5, std of 0.5 and 3 RGB channels
        ])

        # =========================================================== #

        # Load CIFAR-10 training data and test data and apply transformations
        train_data = torchvision.datasets.CIFAR10(root='/data', train=True, transform=transform, download=True)
        test_data = torchvision.datasets.CIFAR10(root='/data', train=False, transform=transform, download=True)

        # Create a DataLoader for training data with batch size of 32, shuffling data, and using 2 threads for loading
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

        # =========================================================== #

        # Retrieve the index 0 image from the training dataset
        image, label = train_data[0]

        # =========================================================== #

        # Print the size of the image tensor
        image.size()

        # =========================================================== #

        # The class names for the CIFAR-10 (10 classes)
        class_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # =========================================================== #

        # Defining the architecture of a Convolutional Neural Network (CNN)
        class NeuralNet(nn.Module):

            def __init__(self):
                super().__init__()

                # Making the convolutional layer
                # """Can change number of feature map, kernel size, max pooling size, different channels, number of neurons but keep compatible"""
                self.conv1 = nn.Conv2d(3, 12, 5) # (12 channels, 28 pixels, 28 pixels)
                self.pool = nn.MaxPool2d(2, 2) # 2 x 2 pixels and will create 1 pixel out of it (12, 14, 14)
                self.conv2 = nn.Conv2d(12, 24, 5) # (24, 10, 10) -> (24, 5, 5) -> Flatten (24 * 5 * 5)
                # Fully connected layers
                self.fc1 = nn.Linear(24 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10) 

            # """All this will be random in the beginning but will be trained as time goes by"""
            def forward(self, x): # Applies these layers on the output
                # Define the forward pass (data flow through the layers)
                x = self.pool(F.relu(self.conv1(x))) # F.relu Breaks linearity
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        # =========================================================== #

        # if __name__ == '__main__':  # Add this guard to prevent multiprocessing issues

            # """" ONLY UNCOMMENT THIS CODE IF YOU WANT TO TRAIN THE MODEL AGAIN """

            # net = NeuralNet()
            # loss_function = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            # # Training it
            # for epoch in range(5):
            #     print(f'Training epoch {epoch}...')

            #     running_loss = 0.0

            #     for i, data in enumerate(train_loader):
            #         inputs, labels = data

            #         optimizer.zero_grad() # Reset gradients

            #         outputs = net(inputs)

            #         loss = loss_function(outputs, labels)
            #         loss.backward()
            #         optimizer.step()

            #         running_loss += loss.item()

            #     print(f'Loss: {running_loss / len(train_loader):.4f}')

            # torch.save(net.state_dict(), 'trained_net.pth')

            # =========================================================== #

        # Loading the pre-trained model
        net = NeuralNet()
        net.load_state_dict(torch.load('trained_net.pth')) # Saving the model parameters in the pth file

        # =========================================================== #

        # Counter for correct predictions and total test images
        correct = 0
        total = 0

        net.eval()

        # Disable gradient calculation during evaluation
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(f'Accuracy: {accuracy}%')

        # =========================================================== #

        if submitted_image is not None:  # Ensure submitted_image has been set
            new_transform = transforms.Compose([ # Define transformations
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            def load_image(image):
                image = new_transform(image)
                image = image.unsqueeze(0)
                return image
            
            # List to store transformed images
            images = [load_image(submitted_image)]

            net.eval() # Set the model to evaluation mode
            predicted_value = ''
            player_res = []
            with torch.no_grad(): # Disable gradient calculation for prediction
                for image in images:
                    output = net(image)
                    _, predicted = torch.max(output, 1)
                    print(f'Prediction: {class_name[predicted.item()]}')
                    predicted_value = class_name[predicted.item()]
                    player_res.append(predicted_value)
                    # st.write(predicted_value)
            for res in player_res:
                st.write(f'The players drawing is probably a {res}')        


if __name__ == '__main__':
    main()