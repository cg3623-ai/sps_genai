# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy the pyproject.toml file and install dependencies using uv
COPY pyproject.toml /code/
RUN uv sync

# Download the spaCy model used by the /embed endpoint
# (en_core_web_md ships 300-dimensional word vectors; do NOT use en_core_web_sm)
RUN uv run python -m spacy download en_core_web_md

# Copy the application code
COPY ./app /code/app

# Command to run the application
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
