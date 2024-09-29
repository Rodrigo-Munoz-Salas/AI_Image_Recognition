# AI Image Recognition
This project implements a CNN trained by CIFAR/10 in order to implement image recognition to create a telephone like game. The players draw the images and the machine attemps to identify the image, passing it on to the next user.

# Safe Software Supply Chain
This project uses the PyTorch Chainguard image as a base image. This image is free of vulnerabilities (zero CVEs), guaranteeing that external actuators cannot exploit vulnerabilities. However, two dependencies were added for the development of the GUI: streamlit and streamlit-drawable-canvas

# How to Run AI Image Recognition
```bash
# Clone the repository
git clone https://github.com/Rodrigo-Munoz-Salas/AI_Image_Recognition.git

# Build the Docker image (from Chainguard images)
docker build . -t "ai-recognition-image"

# Run the docker container
docker run -d -p 8501:8501 quick-draw-image

```
Go to your local host at 8501 port: localhost:8501

# How to Interact with the AI Image Recognition
Select the **number of players**, click on **Start**, and begin drawing!
Once you finish your drawing, click on **Submit Drawing** and wait for the program to give you its prediction!

# How to Stop AI Image Recognition
```bash
# Stop the Docker container
docker stop CONTAINER_ID

```
