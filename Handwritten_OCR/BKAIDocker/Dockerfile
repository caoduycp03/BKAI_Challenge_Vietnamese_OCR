FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip sudo

COPY . /app

RUN pip3 install -r requirements.txt






