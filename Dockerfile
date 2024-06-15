# No idea if thi works correctly at all rn

FROM python:3.10

RUN apt-get update

RUN echo python --version
RUN echo pip --version

ENV APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

# determines where we'll work inside the docker container
# kind of like running mkdir and cd concurrently
WORKDIR $APP_FOLDER

COPY . $APP_FOLDER

RUN pip install -e .[all]
