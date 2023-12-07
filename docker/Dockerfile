FROM pytorch/pytorch

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends \
    curl wget build-essential openjdk-17-jdk git \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV LUCENE_VERSION=9.7.0
ENV ANT_VERSION=1.10.14
ENV PATH=$PATH:/pybool_ir/ant/bin:/opt/conda/bin
ARG MIRROR=downloads
ARG JCC_JDK=${JAVA_HOME}
ARG JCC_ARGSEP=";"
ARG JCC_LFLAGS="-L$JAVA_HOME/lib;-ljava;-L$JAVA_HOME/lib/server;-ljvm;-Wl,-rpath=$JAVA_HOME/lib:$JAVA_HOME/lib/server"
ARG JCC="python -m jcc --shared --wheel"
RUN git clone https://github.com/fschlatt/pybool_ir
WORKDIR /workspace/pybool_ir
RUN git checkout neural
RUN curl -O https://${MIRROR}.apache.org/lucene/pylucene/pylucene-${LUCENE_VERSION}-src.tar.gz \
    && gunzip pylucene-${LUCENE_VERSION}-src.tar.gz \
    && tar -xvf pylucene-${LUCENE_VERSION}-src.tar \
    && rm pylucene-${LUCENE_VERSION}-src.tar \
    && ln -s pylucene-${LUCENE_VERSION} pylucene \
    && curl -O https://${MIRROR}.apache.org/ant/binaries/apache-ant-${ANT_VERSION}-bin.tar.gz \
    && gunzip apache-ant-${ANT_VERSION}-bin.tar.gz \
    && tar -xvf apache-ant-${ANT_VERSION}-bin.tar \
    && rm apache-ant-${ANT_VERSION}-bin.tar \
    && ln -s apache-ant-${ANT_VERSION} ant
RUN cd pylucene/jcc \
    && python setup.py build \
    && python setup.py install
RUN cd pylucene \
    && NUM_FILES=10 make
RUN pip install pylucene/dist/*.whl lupyne parsedatetime progressbar2 appdirs transformers ir_datasets ir_measures pyparsing