FROM ubuntu:xenial
#FROM continuumio/anaconda:latest


ARG PYTHON_VERSION=3.6


ENV PROGRAM_ROOT /opt/program


RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         awscli \
         nginx \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libglib2.0-0 \
         libsm6 \
         libxext6 \
         libfontconfig1 \
         libxrender1 && \
      rm -rf /var/lib/apt/lists/*



RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION  && \
     /opt/conda/bin/conda clean -ya



ENV PATH /opt/conda/bin:$PATH


RUN pip --no-cache-dir install \
    Cython requests imgaug pillow wget boto3 flask gevent gunicorn jupyter

RUN pip --no-cache-dir install \
    torch==0.4.1 torchvision==0.2.1 pycocotools==2.0.0 opencv-python==3.4.0.14 numpy==1.15.0

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"


COPY pose $PROGRAM_ROOT
WORKDIR $PROGRAM_ROOT

