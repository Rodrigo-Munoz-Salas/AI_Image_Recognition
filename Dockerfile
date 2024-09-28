FROM cgr.dev/chainguard/pytorch:latest
COPY main.py ./
ENTRYPOINT ["python", "main.py"]