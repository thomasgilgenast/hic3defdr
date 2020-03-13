# dep builder: builds wheels for all deps
FROM python:2.7 AS dep-builder

# build wheels for all deps in requirements.txt
COPY requirements.txt /build/requirements.txt
RUN pip wheel --no-deps -w /build/dist -r /build/requirements.txt

# base image: installs wheels for all dependencies
FROM python:2.7-slim AS base

# add wget to download data e.g. for tutorial
RUN apt-get update && apt-get install -y wget && apt-get clean

# matplotlib configuration
ENV MPLBACKEND=agg

# copy all wheels from builder and install
COPY --from=dep-builder [ "/build/dist/*.whl", "/install/" ]
RUN pip install --no-index /install/*.whl \
    && rm -rf /install

# app builder: builds wheel for just our app
FROM python:2.7 as app-builder

# copy app code and build wheel for our app
# the only file we've used above this line is requirements.txt
# therefore only the lines below will be re-run when app code changes
ARG VERSION=unknown
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION
COPY . /build
RUN pip wheel --no-deps -w /build/dist /build

# final image: start from base and add just our app
FROM base AS final

# copy all wheels from builder and install
COPY --from=app-builder [ "/build/dist/*.whl", "/install/" ]
RUN pip install --no-index /install/*.whl \
    && rm -rf /install

ENTRYPOINT bash
