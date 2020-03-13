FROM python:2.7

# add wget to download data e.g. for tutorial
RUN apt-get update && apt-get install -y wget && apt-get clean

# install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# matplotlib configuration
ENV MPLBACKEND=agg

# copy and install app code
ARG VERSION=unknown
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION
COPY . /app
RUN pip install --no-deps /app

ENTRYPOINT bash
