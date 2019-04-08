FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get -y install curl

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.0-linux-x86_64.tar.gz
RUN tar xvfa julia-1.0.0-linux-x86_64.tar.gz

COPY mlp.jl /julia-1.0.0/bin/mlp.jl
COPY iterate_benchmarks.jl /julia-1.0.0/bin/iterate_benchmarks.jl

WORKDIR /julia-1.0.0/bin
CMD ["./julia"]