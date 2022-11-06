from tensorflow/tensorflow:latest-gpu

ARG DEBIAN_FRONTEND=noninteractive

RUN groupadd karolis -g 1000  && useradd -u 1000 -d /home/karolis -m --shell /bin/bash --no-user-group karolis && usermod -g 1000 karolis

ENV LANG C.UTF-8
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8

COPY ./personal_configs/inputrc /home/karolis/.inputrc
COPY ./personal_configs/vimrc /home/karolis/.vimrc

RUN apt-get update && apt-get install vim less -y

RUN python -m pip install matplotlib Cython scipy pandas keras h5py hdfs scikit-learn opencv-python-headless seaborn IPython 

RUN python -m pip install lightgbm

#RUN python -m pip install tensorflow_model_analysis

