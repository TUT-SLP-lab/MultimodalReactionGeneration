ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.04-py3
FROM ${BASE_IMAGE}
# ARG FFMPEG_IMAGE=jrottenberg/ffmpeg
# FROM ${FFMPEG_IMAGE}

# for CUDA 12
ARG OPENCV_VERSION=4.7.0
ARG PYTHON_VERSION=3.8

# for installing ffmpeg & opencv into docker container
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && \
    apt-get -y install ffmpeg
#     apt install -y libopencv-dev

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
    vim unzip byobu wget tree git cmake && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# install opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev libgtk-3-dev \
    libavformat-dev libswscale-dev libhdf5-serial-dev qt5-default \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libopenblas-dev libgflags-dev &&\
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN wget -c https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz
RUN tar -zxvf ${OPENCV_VERSION}.tar.gz && rm ${OPENCV_VERSION}.tar.gz
RUN mkdir /workspace/opencv-${OPENCV_VERSION}/build

## opencv_contrib
RUN wget -c https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz &&\
    tar -zxvf ${OPENCV_VERSION}.tar.gz && rm /workspace/${OPENCV_VERSION}.tar.gz
WORKDIR /workspace/opencv-${OPENCV_VERSION}/build

## make opencv
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python${PYTHON_VERSION}/dist-packages \
    -D WITH_TBB=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D BUILD_opencv_cudacodec=OFF \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN="7.5, 8.0" \
    -D WITH_V4L=ON \
    -D WITH_QT=ON \
    -D WITH_OPENGL=ON \
    -D WITH_GSTREAMER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib-${OPENCV_VERSION}/modules \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D BUILD_EXAMPLES=OFF .. &&\
    make -j20 && make install && ldconfig

# install pip 
RUN pip install --upgrade pip && \
    pip install audioread==3.0.0 \
    dfcon \
    joblib==1.3.2 \
    librosa==0.10.1 \
    matplotlib==3.7.3 \
    numpy==1.24.4 \
    opencv-python==4.8.0.74 \
    opencv-contrib-python==4.8.0.74 \
    soundfile==0.12.1 \
    mediapipe \
    pyannote.audio \
    pyworld 

WORKDIR /home/MultimodalReactionGeneration
CMD bash -c "npm start && /bin/bash"