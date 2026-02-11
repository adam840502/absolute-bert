FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime AS builder

# # Install Poetry (no venv mode)
# ARG POETRY_VERSION=1.8.3
# RUN pip install --no-cache-dir poetry==$POETRY_VERSION && \
#     poetry config virtualenvs.create false

WORKDIR /workspace

RUN apt-get update && apt-get install -y build-essential

COPY . .

RUN pip install ".[sweep]" --no-cache-dir


# Default run (change to your entry point)
CMD ["bash"]
