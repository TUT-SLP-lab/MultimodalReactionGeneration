ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.04-py3
FROM ${BASE_IMAGE}

RUN pip install audioread==3.0.0
RUN pip install dfcon
RUN pip install joblib==1.3.2
RUN pip install librosa==0.10.1
RUN pip install matplotlib==3.7.3
RUN pip install numpy==1.24.4
RUN pip install opencv-python==4.8.0.76
RUN pip install soundfile==0.12.1

WORKDIR /home/MultimodalReactionGeneration
COPY ../setup.py /home/MultimodalReactionGeneration
COPY ../mr_gen /home/MultimodalReactionGeneration
RUN python setup.py install