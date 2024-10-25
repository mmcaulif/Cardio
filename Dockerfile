FROM python:3.11

# set a directory for the app
WORKDIR /usr/src/app

# set up poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

# set up environment
COPY . .
RUN poetry install --with dev

# Tensorboard
# EXPOSE 0000
