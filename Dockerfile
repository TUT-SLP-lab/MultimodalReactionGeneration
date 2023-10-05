ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.04-py3
FROM ${BASE_IMAGE}


RUN pip install --upgrade pip && \
    pip install audioread==3.0.0 \
    dfcon \
    joblib==1.3.2 \
    librosa==0.10.1 \
    matplotlib==3.7.3 \
    numpy==1.24.4 \
    opencv-python==4.8.0.74 \
    soundfile==0.12.1

WORKDIR /home/MultimodalReactionGeneration
CMD bash -c "npm start && /bin/bash"
# COPY ./setup.py .
# COPY ./mr_gen .
# RUN python setup.py install
# RUN python -c "from mr_gen.utils.video import Video"