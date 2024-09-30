# Using pytorch-dev chainguard image
FROM cgr.dev/chainguard/pytorch:latest-dev

USER root

# Copy the files into the docker repo
COPY main.py ./
COPY trained_net.pth ./
COPY views/ ./views/

# Install streamlit package and set path environment
RUN python -m venv /app/venv
RUN pip install --no-cache-dir streamlit streamlit-drawable-canvas
ENV PATH="$PATH:/app/venv/bin"

# entrypoint to run the app
ENTRYPOINT ["streamlit", "run", "main.py"]

# Expose the default Streamlit port
EXPOSE 8501