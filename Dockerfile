FROM jupyter/scipy-notebook
LABEL maintainer="Marco Pleines <m.pleines@devbeyond.de>"

# Install CUDA Toolkit 8.0 runtime and cudnn6
# Source: https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/8.0/runtime/cudnn6/Dockerfile
USER root
RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list

ENV CUDA_VERSION 8.0.61

ENV CUDA_PKG_VERSION 8-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvrtc-$CUDA_PKG_VERSION \
        cuda-nvgraph-$CUDA_PKG_VERSION \
        cuda-cusolver-$CUDA_PKG_VERSION \
        cuda-cublas-8-0=8.0.61.2-1 \
        cuda-cufft-$CUDA_PKG_VERSION \
        cuda-curand-$CUDA_PKG_VERSION \
        cuda-cusparse-$CUDA_PKG_VERSION \
        cuda-npp-$CUDA_PKG_VERSION \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-8.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=8.0"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.21
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn6=$CUDNN_VERSION-1+cuda8.0 && \
    rm -rf /var/lib/apt/lists/*

# Install Tensorflow and Keras
USER $NB_USER

RUN conda install --quiet --yes \
    'tensorflow-gpu=1.4*' \
    'keras=2.0*' && \
    conda clean -tipsy && \
fix-permissions $CONDA_DIR

# Install pip
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	&& \
	curl -O https://bootstrap.pypa.io/get-pip.py && \
    	python get-pip.py && \
	rm get-pip.py

# Install Jupyter Tensorboard Plugin
USER $NB_USER
RUN pip install jupyter-tensorboard

# Retrieve and prepare ML-Agents Python files
USER root
ADD https://github.com/Unity-Technologies/ml-agents/archive/0.2.1d.tar.gz /home/jovyan/
RUN tar -xzvf /home/jovyan/0.2.1d.tar.gz -C /home/jovyan/ \
	&& rm /home/jovyan/0.2.1d.tar.gz \
	&& mv /home/jovyan/ml-agents-0.2.1d/python/ /home/jovyan/temp \
	&& rm -r /home/jovyan/ml-agents-0.2.1d/ \
	&& mv /home/jovyan/temp /home/jovyan/ml-agents-0.2.1d  \
	&& pip install /home/jovyan/ml-agents-0.2.1d/. \
	&& rm -rf work \
	&& fix-permissions /home/jovyan/ml-agents-0.2.1d

# Add usefull files
COPY Troubleshooting_Utilities.ipynb /home/jovyan
# Fix permissions
RUN chmod -R 777 /home/jovyan/Troubleshooting_Utilities.ipynb
# Override Jupyter Config
COPY jupyter_notebook_config.py /etc/jupyter/

USER $NB_USER
