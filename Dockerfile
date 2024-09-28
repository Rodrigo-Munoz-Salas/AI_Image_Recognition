FROM cgr.dev/chainguard/python:latest-dev
COPY main.py ./
ENTRYPOINT ["python", "main.py"]