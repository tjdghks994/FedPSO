FROM tensorflow/tensorflow:latest-gpu

ADD . /fl
WORKDIR /fl

RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
RUN python -m pip install tensorflow-federated

