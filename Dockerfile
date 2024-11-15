FROM python:3.11-slim

# set a directory for the app
WORKDIR /app

# set up poetry
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN pip install poetry
RUN poetry config virtualenvs.create false

# set up environment
COPY . /app
RUN poetry install --with dev

# Tensorboard
EXPOSE 6006

# https://stackoverflow.com/questions/41523005/how-to-use-tensorboard-in-a-docker-container-on-windows
