FROM cgr.dev/chainguard/pytorch:latest-dev
COPY main.py ./
ENTRYPOINT ["python", "main.py"]