
Multi-Layer Perceptron Classification on GPU with Julia
====

**Algorithm:** Multi-layer perceptron neural network classifier with the [Flux](https://github.com/FluxML) ML library

**Task:** Compare loss and accuracy of the algorithm over epochs on different computing platforms, using CPUs or GPUs


```julia
macro bash_str(s) open(`bash`,"w",stdout) do io; print(io, s); end; end; # this creates a bash macro
```

Write code to create, train and benchmark a MLP NN
---

### Code below modified from [FluxML/model-zoo](https://github.com/FluxML/model-zoo/blob/master/vision/mnist/mlp.jl).

This version will run on CPU because we are not ```using CuArrays```, meaning the ```|> gpu``` will be ignored.


```julia
module MLPClassifier

    using Flux, Statistics
    using Flux: onehotbatch, onecold, crossentropy, throttle # Flux is a neural network machine learning library, rivals TensorFlow
    using Base.Iterators: repeated
#     using CuArrays

    function create_model()
        
        m = Chain(Dense(28^2, 32, relu), Dense(32, 10), softmax) |> gpu
        return m
    
    end

    function benchmark_model(m, imgs, labels; epochs=3, dataset_n=1)

        # Stack images into one large batch. Concatenates along 2 dimensions
        X = hcat(float.(reshape.(imgs, :))...) |> gpu # pipe to gpu, this does nothing when CuArrays is not loaded

        # One-hot-encode the labels
        Y = onehotbatch(labels, 0:9) |> gpu     

        loss(x, y) = crossentropy(m(x), y)

        accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

        # Create a dataset by repeating dataset_n times
        dataset = repeated((X, Y), dataset_n)

        # accuracy() computes the fraction of correctly predicted outcomes in outputs (Y) according to the given true targets (X).
        # loss() the loss function gives a number which an optimization would seek to minimize

        opt = ADAM()

        # Train the multi-layer-perceptron:
        start_time = time_ns()
        for i = 1:epochs
            Flux.train!(loss, params(m), dataset, opt)
        end
        end_time = time_ns()

        # Results
        training_time = (end_time - start_time)/1.0e9 #seconds
        loss_result = loss(X, Y)
        accuracy_result = accuracy(X, Y)

        # Create results dictionary and print to output
        output_dict = Dict("training_time" => training_time, "loss_result" => loss_result, "accuracy_result" => accuracy_result)
        return output_dict

    end

end;
```


```julia
write("mlp.jl", In[IJulia.n-1]); # write the previously run cell to file
```

### Test the model and benchmark functions work:


```julia
using Flux.Data.MNIST # MNIST digits
model = MLPClassifier.create_model()
```




    Chain(Dense(784, 32, NNlib.relu), Dense(32, 10), NNlib.softmax)




```julia
Main.MLPClassifier.benchmark_model(model, MNIST.images(), MNIST.labels(), epochs=2, dataset_n=1)
```




    Dict{String,Real} with 3 entries:
      "accuracy_result" => 0.140633
      "training_time"   => 5.8175
      "loss_result"     => 2.24282 (tracked)



Create a script to run benchmarks of the model
----

Here we get the median result for the benchmarks from multiple ```repeats```, then print:


```julia
include("./mlp.jl")
using Main.MLPClassifier
using Statistics
using Flux.Data.MNIST

# Get benchmarking paramaters:
repeats = 3 # defaults
epochs = 1
dataset_n = 1

if length(ARGS) == 3 # replace parameters with command line args when provided

    repeats = parse(Int64, ARGS[1])
    epochs = parse(Int64, ARGS[2])
    dataset_n = parse(Int64, ARGS[3])
    
end

# Create model:
model = MLPClassifier.create_model()

# Benchmark the model:
accuracy_results = []
training_times = []
loss_results = []

imgs = MNIST.images()
labels = MNIST.labels()

for i = 1:repeats
    benchmarks = Main.MLPClassifier.benchmark_model(model, imgs, labels, epochs=epochs, dataset_n=dataset_n)
    append!(accuracy_results, benchmarks["accuracy_result"])
    append!(training_times, benchmarks["training_time"])
    append!(loss_results, benchmarks["loss_result"])
end

# Print benchmarks
print(Dict("training_time" => median(training_times), "loss_result" => median(loss_results), "accuracy_result" => median(accuracy_results)))
```

    Dict{String,Real}(

    WARNING: replacing module MLPClassifier.


    "accuracy_result"=>0.15815,"training_time"=>0.455815,"loss_result"=>2.28231 (tracked))


```julia
write("iterate_benchmarks.jl", In[IJulia.n-1]) # write the previously run cell to file
```




    1011



Build a Docker container running the benchmarks and push to Docker Hub
---

### The Dockerfile

I use a CUDA base image that will allow for GPU functionality. For now, I refrain from installing the ```CuArrays``` Julia package.

I have set the benchmarks to take command line arguments which can be changed when we run the container.


```julia
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
RUN ./julia -e 'using Pkg; Pkg.add("Flux")'
# RUN ./julia -e 'using Pkg; Pkg.add("CuArrays")'
CMD ./julia iterate_benchmarks.jl 1 1 1
```


    syntax: extra token "nvidia" after end of expression

    



```julia
write("Dockerfile", In[IJulia.n-1]) # write the previously run cell to file
```




    583



### Build #1
Lets tag this build as ```cpu``` since we are not using ```CuArrays```


```julia
bash"""
docker build -t edwardchalstrey/mlp_classifier:cpu .
"""
```

    Sending build context to Docker daemon  155.6kB
    Step 1/11 : FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
     ---> f722eab170b7
    Step 2/11 : RUN  apt-get update   && apt-get install -y wget   && rm -rf /var/lib/apt/lists/*
     ---> Using cache
     ---> 6c8df59a2db7
    Step 3/11 : RUN apt-get update
     ---> Using cache
     ---> 52daa6d5e08f
    Step 4/11 : RUN apt-get -y install curl
     ---> Using cache
     ---> f4b4ff210f19
    Step 5/11 : RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.0-linux-x86_64.tar.gz
     ---> Using cache
     ---> 9b5d7c5c2cfd
    Step 6/11 : RUN tar xvfa julia-1.0.0-linux-x86_64.tar.gz
     ---> Using cache
     ---> 71f8c0aebc04
    Step 7/11 : COPY mlp.jl /julia-1.0.0/bin/mlp.jl
     ---> dbc821396ac6
    Step 8/11 : COPY iterate_benchmarks.jl /julia-1.0.0/bin/iterate_benchmarks.jl
     ---> 9565297b2e16
    Step 9/11 : WORKDIR /julia-1.0.0/bin
     ---> Running in e7fbe7fd4f01
    Removing intermediate container e7fbe7fd4f01
     ---> b961383779f7
    Step 10/11 : RUN ./julia -e 'using Pkg; Pkg.add("Flux")'
     ---> Running in 5f444479957c
       Cloning default registries into /root/.julia/registries
       Cloning registry General from "https://github.com/JuliaRegistries/General.git"
    [2K[?25h  Updating registry at `~/.julia/registries/General`2 %0.0 %
      Updating git-repo `https://github.com/JuliaRegistries/General.git`
    [?25l[2K[?25h Resolving package versions...
     Installed DiffRules ──────────── v0.0.10
     Installed CSTParser ──────────── v0.5.2
     Installed MacroTools ─────────── v0.5.0
     Installed BinaryProvider ─────── v0.5.3
     Installed DataStructures ─────── v0.15.0
     Installed Flux ───────────────── v0.8.2
     Installed Adapt ──────────────── v0.4.2
     Installed Colors ─────────────── v0.9.5
     Installed Media ──────────────── v0.5.0
     Installed NNlib ──────────────── v0.5.0
     Installed ZipFile ────────────── v0.8.1
     Installed CodecZlib ──────────── v0.5.2
     Installed FixedPointNumbers ──── v0.5.3
     Installed Requires ───────────── v0.5.2
     Installed Compat ─────────────── v2.1.0
     Installed URIParser ──────────── v0.4.0
     Installed Missings ───────────── v0.4.0
     Installed ForwardDiff ────────── v0.10.3
     Installed SortingAlgorithms ──── v0.3.1
     Installed DiffResults ────────── v0.0.4
     Installed CommonSubexpressions ─ v0.2.0
     Installed Tracker ────────────── v0.1.0
     Installed NaNMath ────────────── v0.3.2
     Installed Tokenize ───────────── v0.5.3
     Installed Reexport ───────────── v0.2.0
     Installed Juno ───────────────── v0.7.0
     Installed SpecialFunctions ───── v0.7.2
     Installed ColorTypes ─────────── v0.7.5
     Installed AbstractTrees ──────── v0.2.1
     Installed TranscodingStreams ─── v0.9.3
     Installed StatsBase ──────────── v0.29.0
     Installed BinDeps ────────────── v0.8.10
     Installed OrderedCollections ─── v1.0.2
     Installed StaticArrays ───────── v0.10.3
      Updating `~/.julia/environments/v1.0/Project.toml`
      [587475ba] + Flux v0.8.2
      Updating `~/.julia/environments/v1.0/Manifest.toml`
      [1520ce14] + AbstractTrees v0.2.1
      [79e6a3ab] + Adapt v0.4.2
      [9e28174c] + BinDeps v0.8.10
      [b99e7846] + BinaryProvider v0.5.3
      [00ebfdb7] + CSTParser v0.5.2
      [944b1d66] + CodecZlib v0.5.2
      [3da002f7] + ColorTypes v0.7.5
      [5ae59095] + Colors v0.9.5
      [bbf7d656] + CommonSubexpressions v0.2.0
      [34da2185] + Compat v2.1.0
      [864edb3b] + DataStructures v0.15.0
      [163ba53b] + DiffResults v0.0.4
      [b552c78f] + DiffRules v0.0.10
      [53c48c17] + FixedPointNumbers v0.5.3
      [587475ba] + Flux v0.8.2
      [f6369f11] + ForwardDiff v0.10.3
      [e5e0dc1b] + Juno v0.7.0
      [1914dd2f] + MacroTools v0.5.0
      [e89f7d12] + Media v0.5.0
      [e1d29d7a] + Missings v0.4.0
      [872c559c] + NNlib v0.5.0
      [77ba4419] + NaNMath v0.3.2
      [bac558e1] + OrderedCollections v1.0.2
      [189a3867] + Reexport v0.2.0
      [ae029012] + Requires v0.5.2
      [a2af1166] + SortingAlgorithms v0.3.1
      [276daf66] + SpecialFunctions v0.7.2
      [90137ffa] + StaticArrays v0.10.3
      [2913bbd2] + StatsBase v0.29.0
      [0796e94c] + Tokenize v0.5.3
      [9f7883ad] + Tracker v0.1.0
      [3bb67fe8] + TranscodingStreams v0.9.3
      [30578b45] + URIParser v0.4.0
      [a5390f91] + ZipFile v0.8.1
      [2a0f44e3] + Base64 
      [ade2ca70] + Dates 
      [8bb1440f] + DelimitedFiles 
      [8ba89e20] + Distributed 
      [b77e0a4c] + InteractiveUtils 
      [76f85450] + LibGit2 
      [8f399da3] + Libdl 
      [37e2e46d] + LinearAlgebra 
      [56ddb016] + Logging 
      [d6f4376e] + Markdown 
      [a63ad114] + Mmap 
      [44cfe95a] + Pkg 
      [de0858da] + Printf 
      [9abbd945] + Profile 
      [3fa0cd96] + REPL 
      [9a3f8284] + Random 
      [ea8e919c] + SHA 
      [9e88b42a] + Serialization 
      [1a1011a3] + SharedArrays 
      [6462fe0b] + Sockets 
      [2f01184e] + SparseArrays 
      [10745b16] + Statistics 
      [8dfed614] + Test 
      [cf7118a7] + UUIDs 
      [4ec0a83e] + Unicode 
      Building ZipFile ─────────→ `~/.julia/packages/ZipFile/YHTbb/deps/build.log`
      Building CodecZlib ───────→ `~/.julia/packages/CodecZlib/9jDi1/deps/build.log`
      Building SpecialFunctions → `~/.julia/packages/SpecialFunctions/fvheQ/deps/build.log`
    Removing intermediate container 5f444479957c
     ---> 4e94c0356a83
    Step 11/11 : CMD ./julia iterate_benchmarks.jl 1 1 1
     ---> Running in 09bf55be809a
    Removing intermediate container 09bf55be809a
     ---> 760996b36a7c
    Successfully built 760996b36a7c
    Successfully tagged edwardchalstrey/mlp_classifier:cpu


### Lets check we can run the container


```julia
bash"""
docker run edwardchalstrey/mlp_classifier:cpu
"""
```

    [ Info: Downloading MNIST dataset
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   469  100   469    0     0    174      0  0:00:02  0:00:02 --:--:--   174
    100 9680k  100 9680k    0     0  2113k      0  0:00:04  0:00:04 --:--:-- 9008k
    [ Info: Downloading MNIST dataset
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   469  100   469    0     0   1104      0 --:--:-- --:--:-- --:--:--  1103
    100 28881  100 28881    0     0  30340      0 --:--:-- --:--:-- --:--:-- 30340
    [ Info: Downloading MNIST dataset
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   467  100   467    0     0   1123      0 --:--:-- --:--:-- --:--:--  1122
    100 1610k  100 1610k    0     0  1143k      0  0:00:01  0:00:01 --:--:-- 4686k
    [ Info: Downloading MNIST dataset
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   467  100   467    0     0   1130      0 --:--:-- --:--:-- --:--:--  1130
    100  4542  100  4542    0     0   5426      0 --:--:-- --:--:-- --:--:-- 21125


    Dict{String,Real}("accuracy_result"=>0.0845667,"training_time"=>6.81843,"loss_result"=>2.31553 (tracked))

### Then push to Docker Hub


```julia
bash"""
docker push edwardchalstrey/mlp_classifier:cpu
"""
```

    The push refers to repository [docker.io/edwardchalstrey/mlp_classifier]
    0068878821ab: Preparing
    34e6dc24873a: Preparing
    6a425a0659f9: Preparing
    c16a6ae73666: Preparing
    0d6fbe8a52c0: Preparing
    371ecab57b6d: Preparing
    1b9b09744ade: Preparing
    9ad6d222ddc9: Preparing
    8c1e86448329: Preparing
    c797737f624c: Preparing
    37f8e8828549: Preparing
    36382f64a35d: Preparing
    5e57e1e34e26: Preparing
    889ba48cb5a1: Preparing
    68dda0c9a8cd: Preparing
    f67191ae09b8: Preparing
    b2fd8b4c3da7: Preparing
    0de2edf7bff4: Preparing
    36382f64a35d: Waiting
    5e57e1e34e26: Waiting
    889ba48cb5a1: Waiting
    68dda0c9a8cd: Waiting
    f67191ae09b8: Waiting
    b2fd8b4c3da7: Waiting
    371ecab57b6d: Waiting
    1b9b09744ade: Waiting
    9ad6d222ddc9: Waiting
    8c1e86448329: Waiting
    0de2edf7bff4: Waiting
    c797737f624c: Waiting
    37f8e8828549: Waiting
    0d6fbe8a52c0: Layer already exists
    c16a6ae73666: Layer already exists
    371ecab57b6d: Layer already exists
    1b9b09744ade: Layer already exists
    9ad6d222ddc9: Layer already exists
    8c1e86448329: Layer already exists
    34e6dc24873a: Pushed
    c797737f624c: Layer already exists
    37f8e8828549: Layer already exists
    6a425a0659f9: Pushed
    36382f64a35d: Layer already exists
    5e57e1e34e26: Layer already exists
    889ba48cb5a1: Layer already exists
    f67191ae09b8: Layer already exists
    68dda0c9a8cd: Layer already exists
    b2fd8b4c3da7: Layer already exists
    0de2edf7bff4: Layer already exists
    0068878821ab: Pushed
    cpu: digest: sha256:0a394b64ba529fd9640c9763e5302cf10c79db72da6151904cb8f987c1f0976c size: 4101


Now lets create a version of the code and Docker container where the model is running on NVIDIA GPU with CUDA
---

To run in a Docker container on a machine with NVIDIA GPUs, the following steps must be taken:

1. Follow the [installation instructions for CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (can download from [here](https://developer.nvidia.com/cuda-toolkit)), then the [post-installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#mandatory-post) and make sure you have a version of Docker that [is supported](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#which-docker-packages-are-supported)
2. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)#installing-version-20)
3. Run the container with nvidia-docker e.g. ```nvidia-docker run edwardchalstrey/mlp_classifier:gpu```

### First lets un-comment CuArrays in the classifier

This version, where we are ```using CuArrays``` can't be run without NVIDIA GPU support:


```julia
module MLPClassifier

    using Flux, Statistics
    using Flux: onehotbatch, onecold, crossentropy, throttle # Flux is a neural network machine learning library, rivals TensorFlow
    using Base.Iterators: repeated
    using CuArrays

    function create_model()
        
        m = Chain(Dense(28^2, 32, relu), Dense(32, 10), softmax) |> gpu
        return m
    
    end

    function benchmark_model(m, imgs, labels; epochs=3, dataset_n=1)

        # Stack images into one large batch. Concatenates along 2 dimensions
        X = hcat(float.(reshape.(imgs, :))...) |> gpu # pipe to gpu, this does nothing when CuArrays is not loaded

        # One-hot-encode the labels
        Y = onehotbatch(labels, 0:9) |> gpu     

        loss(x, y) = crossentropy(m(x), y)

        accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

        # Create a dataset by repeating dataset_n times
        dataset = repeated((X, Y), dataset_n)

        # accuracy() computes the fraction of correctly predicted outcomes in outputs (Y) according to the given true targets (X).
        # loss() the loss function gives a number which an optimization would seek to minimize

        opt = ADAM()

        # Train the multi-layer-perceptron:
        start_time = time_ns()
        for i = 1:epochs
            Flux.train!(loss, params(m), dataset, opt)
        end
        end_time = time_ns()

        # Results
        training_time = (end_time - start_time)/1.0e9 #seconds
        loss_result = loss(X, Y)
        accuracy_result = accuracy(X, Y)

        # Create results dictionary and print to output
        output_dict = Dict("training_time" => training_time, "loss_result" => loss_result, "accuracy_result" => accuracy_result)
        return output_dict

    end

end
```

    WARNING: replacing module MLPClassifier.
    ┌ Info: Precompiling CuArrays [3a865a2d-5b23-5a0f-bc46-62713ec82fae]
    └ @ Base loading.jl:1192
    ERROR: LoadError: LoadError: UndefVarError: CUBLAS not defined
    Stacktrace:
     [1] top-level scope at none:0 (repeats 2 times)
     [2] include at ./boot.jl:317 [inlined]
     [3] include_relative(::Module, ::String) at ./loading.jl:1044
     [4] include at ./sysimg.jl:29 [inlined]
     [5] include(::String) at /Users/echalstrey/.julia/packages/CuArrays/PD3UJ/src/CuArrays.jl:3
     [6] top-level scope at none:0
     [7] include at ./boot.jl:317 [inlined]
     [8] include_relative(::Module, ::String) at ./loading.jl:1044
     [9] include(::Module, ::String) at ./sysimg.jl:29
     [10] top-level scope at none:2
     [11] eval at ./boot.jl:319 [inlined]
     [12] eval(::Expr) at ./client.jl:393
     [13] top-level scope at ./none:3
    in expression starting at /Users/echalstrey/.julia/packages/CuArrays/PD3UJ/src/deprecated.jl:5
    in expression starting at /Users/echalstrey/.julia/packages/CuArrays/PD3UJ/src/CuArrays.jl:53



    Failed to precompile CuArrays [3a865a2d-5b23-5a0f-bc46-62713ec82fae] to /Users/echalstrey/.julia/compiled/v1.0/CuArrays/7YFE0.ji.

    

    Stacktrace:

     [1] error(::String) at ./error.jl:33

     [2] compilecache(::Base.PkgId, ::String) at ./loading.jl:1203

     [3] _require(::Base.PkgId) at ./loading.jl:960

     [4] require(::Base.PkgId) at ./loading.jl:858

     [5] require(::Module, ::Symbol) at ./loading.jl:853



```julia
write("mlp.jl", In[IJulia.n-1]) # write the previously run cell to file
```




    1760



### Un-comment CuArrays in the Dockerfile:


```julia
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
RUN ./julia -e 'using Pkg; Pkg.add("Flux")'
RUN ./julia -e 'using Pkg; Pkg.add("CuArrays")'
CMD ./julia iterate_benchmarks.jl
```


    syntax: extra token "nvidia" after end of expression

    



```julia
write("Dockerfile", In[IJulia.n-1]) # write the previously run cell to file
```




    575



### !! Uh oh !! - We have an issue building this version

We are unable to build this, even on a machine with the CUDA toolkit installed


```julia
bash"""
docker build -t edwardchalstrey/mlp_classifier:gpu .
"""
```

    Sending build context to Docker daemon  163.8kB
    Step 1/12 : FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
     ---> f722eab170b7
    Step 2/12 : RUN  apt-get update   && apt-get install -y wget   && rm -rf /var/lib/apt/lists/*
     ---> Using cache
     ---> 6c8df59a2db7
    Step 3/12 : RUN apt-get update
     ---> Using cache
     ---> 52daa6d5e08f
    Step 4/12 : RUN apt-get -y install curl
     ---> Using cache
     ---> f4b4ff210f19
    Step 5/12 : RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.0-linux-x86_64.tar.gz
     ---> Using cache
     ---> 9b5d7c5c2cfd
    Step 6/12 : RUN tar xvfa julia-1.0.0-linux-x86_64.tar.gz
     ---> Using cache
     ---> 71f8c0aebc04
    Step 7/12 : COPY mlp.jl /julia-1.0.0/bin/mlp.jl
     ---> Using cache
     ---> baa9ce743d0f
    Step 8/12 : COPY iterate_benchmarks.jl /julia-1.0.0/bin/iterate_benchmarks.jl
     ---> Using cache
     ---> 1169e455d3c2
    Step 9/12 : WORKDIR /julia-1.0.0/bin
     ---> Using cache
     ---> 414ab46ba9bb
    Step 10/12 : RUN ./julia -e 'using Pkg; Pkg.add("Flux")'
     ---> Using cache
     ---> 766bb6f8905f
    Step 11/12 : RUN ./julia -e 'using Pkg; Pkg.add("CuArrays")'
     ---> Using cache
     ---> 6385a5678d4a
    Step 12/12 : CMD ./julia iterate_benchmarks.jl
     ---> Using cache
     ---> 2e4e1a69a000
    Successfully built 2e4e1a69a000
    Successfully tagged edwardchalstrey/mlp_classifier:gpu


Un-comment below to push when build works


```julia
bash"""
# docker push edwardchalstrey/mlp_classifier:gpu
"""
```

When attempting to run this version on a system with CUDA installed, I get the following errors:

1. ```ERROR: LoadError: LoadError: UndefVarError: CUBLAS not defined```
2. ```ERROR: LoadError: LoadError: Failed to precompile CuArrays```

[On further investigation](https://discourse.julialang.org/t/trouble-building-docker-container-with-cuarrays/22347), it appears that delayed package installation is the only option.

### Alternative solution, delayed package installation:

This one works by running the container, then installing Flux and CuArrays, then running the benchmarks as follows:

1. ```sudo docker run --runtime=nvidia -it edwardchalstrey/juliagpu /bin/bash```
2. In the container do:
    - ```./julia -e 'using Pkg; Pkg.add("Flux")'```
    - ```./julia -e 'using Pkg; Pkg.add("CuArrays")'```
    - ```./julia iterate_benchmarks.jl 1 1 1``` (subsituting different integer arguments here)
    
I've set this up as a separate container called ```edwardchalstrey/juliagpu```.

Doing it this way means that ```CuArrays``` is installed correctly.

Revise the Dockerfile so we don't attempt to install the Julia packages:


```julia
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
```


    syntax: extra token "nvidia" after end of expression

    



```julia
write("Dockerfile", In[IJulia.n-1]) # write the previously run cell to file
```




    465



### Build and push this alternate container:


```julia
bash"""
docker build -t edwardchalstrey/juliagpu .
"""
```

    Sending build context to Docker daemon  163.3kB
    Step 1/10 : FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
     ---> f722eab170b7
    Step 2/10 : RUN  apt-get update   && apt-get install -y wget   && rm -rf /var/lib/apt/lists/*
     ---> Using cache
     ---> 6c8df59a2db7
    Step 3/10 : RUN apt-get update
     ---> Using cache
     ---> 52daa6d5e08f
    Step 4/10 : RUN apt-get -y install curl
     ---> Using cache
     ---> f4b4ff210f19
    Step 5/10 : RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.0-linux-x86_64.tar.gz
     ---> Using cache
     ---> 9b5d7c5c2cfd
    Step 6/10 : RUN tar xvfa julia-1.0.0-linux-x86_64.tar.gz
     ---> Using cache
     ---> 71f8c0aebc04
    Step 7/10 : COPY mlp.jl /julia-1.0.0/bin/mlp.jl
     ---> Using cache
     ---> baa9ce743d0f
    Step 8/10 : COPY iterate_benchmarks.jl /julia-1.0.0/bin/iterate_benchmarks.jl
     ---> Using cache
     ---> 1169e455d3c2
    Step 9/10 : WORKDIR /julia-1.0.0/bin
     ---> Using cache
     ---> 414ab46ba9bb
    Step 10/10 : CMD ["./julia"]
     ---> Using cache
     ---> dd87a2f49058
    Successfully built dd87a2f49058
    Successfully tagged edwardchalstrey/juliagpu:latest



```julia
bash"""
docker push edwardchalstrey/juliagpu
"""
```

    The push refers to repository [docker.io/edwardchalstrey/juliagpu]
    b3076873de1d: Preparing
    492804de3e77: Preparing
    c16a6ae73666: Preparing
    0d6fbe8a52c0: Preparing
    371ecab57b6d: Preparing
    1b9b09744ade: Preparing
    9ad6d222ddc9: Preparing
    8c1e86448329: Preparing
    c797737f624c: Preparing
    37f8e8828549: Preparing
    36382f64a35d: Preparing
    5e57e1e34e26: Preparing
    889ba48cb5a1: Preparing
    68dda0c9a8cd: Preparing
    f67191ae09b8: Preparing
    b2fd8b4c3da7: Preparing
    0de2edf7bff4: Preparing
    9ad6d222ddc9: Waiting
    5e57e1e34e26: Waiting
    889ba48cb5a1: Waiting
    68dda0c9a8cd: Waiting
    f67191ae09b8: Waiting
    b2fd8b4c3da7: Waiting
    0de2edf7bff4: Waiting
    8c1e86448329: Waiting
    c797737f624c: Waiting
    36382f64a35d: Waiting
    1b9b09744ade: Waiting
    37f8e8828549: Waiting
    c16a6ae73666: Layer already exists
    0d6fbe8a52c0: Layer already exists
    371ecab57b6d: Layer already exists
    492804de3e77: Layer already exists
    b3076873de1d: Layer already exists
    1b9b09744ade: Layer already exists
    9ad6d222ddc9: Layer already exists
    8c1e86448329: Layer already exists
    c797737f624c: Layer already exists
    37f8e8828549: Layer already exists
    5e57e1e34e26: Layer already exists
    36382f64a35d: Layer already exists
    889ba48cb5a1: Layer already exists
    68dda0c9a8cd: Layer already exists
    f67191ae09b8: Layer already exists
    b2fd8b4c3da7: Layer already exists
    0de2edf7bff4: Layer already exists
    latest: digest: sha256:9d94e013280dfbc1fa3083f119da3dccede70a9e81eb939914736b8db87c22ff size: 3889
    31db70909fae: Preparing
    575becc17c68: Preparing
    c16a6ae73666: Preparing
    0d6fbe8a52c0: Preparing
    371ecab57b6d: Preparing
    1b9b09744ade: Preparing
    9ad6d222ddc9: Preparing
    8c1e86448329: Preparing
    c797737f624c: Preparing
    37f8e8828549: Preparing
    36382f64a35d: Preparing
    5e57e1e34e26: Preparing
    889ba48cb5a1: Preparing
    68dda0c9a8cd: Preparing
    f67191ae09b8: Preparing
    b2fd8b4c3da7: Preparing
    0de2edf7bff4: Preparing
    68dda0c9a8cd: Waiting
    f67191ae09b8: Waiting
    b2fd8b4c3da7: Waiting
    37f8e8828549: Waiting
    0de2edf7bff4: Waiting
    36382f64a35d: Waiting
    5e57e1e34e26: Waiting
    889ba48cb5a1: Waiting
    c16a6ae73666: Layer already exists
    0d6fbe8a52c0: Layer already exists
    371ecab57b6d: Layer already exists
    1b9b09744ade: Layer already exists
    9ad6d222ddc9: Layer already exists
    8c1e86448329: Layer already exists
    889ba48cb5a1: Layer already exists
    37f8e8828549: Layer already exists
    36382f64a35d: Layer already exists
    5e57e1e34e26: Layer already exists
    f67191ae09b8: Layer already exists
    b2fd8b4c3da7: Layer already exists
    68dda0c9a8cd: Layer already exists
    0de2edf7bff4: Layer already exists
    c797737f624c: Layer already exists
    575becc17c68: Layer already exists
    31db70909fae: Layer already exists
    noevalcb: digest: sha256:181766058c6b7d820856841ba08f13d4bb38d2f7c15e8c9821fdee002996c524 size: 3889


Results
----

I can now run the benchmarks on any computing platform with the CUDA toolkit and NVIDIA-Docker installed (the CPU version on any platform with Docker).

**Platform:**

1. Azure VM 1: Standard NC6 (6 vcpus, 56 GB memory); Ubuntu 18.04; CUDA 9.0

**Benchmarks:**
1. Benchmark repeats = 10; Epochs = 10; Dataset size = 1
2. Benchmark repeats = 10; Epochs = 20; Dataset size = 1
3. Benchmark repeats = 10; Epochs = 50; Dataset size = 1
4. Benchmark repeats = 10; Epochs = 200; Dataset size = 1


```julia
using DataFrames
```

    loaded



```julia
loss_results = [0.4649, 0.480444, 0.267284, 0.258824, 0.131131, 0.141972, NaN, 0.0156178]
accuracy_results = [0.880783, 0.876758, 0.925517, 0.927475, 0.962342, 0.960758, -, 0.997875]
training_times = [3.54387, 0.255519, 9.0776, 0.512827, 18.0053, 2.46613, -, 11.1517]
benchmarks = ["Azure VM 1; CPU; Benchmark 1", "Azure VM 1; GPU; Benchmark 1", "Azure VM 1; CPU; Benchmark 2", "Azure VM 1; GPU; Benchmark 2", "Azure VM 1; CPU; Benchmark 3", "Azure VM 1; GPU; Benchmark 3", "Azure VM 1; CPU; Benchmark 4", "Azure VM 1; GPU; Benchmark 4"]
df = DataFrame(Benchmark = benchmarks, Accuracy = accuracy_results, Loss = loss_results, trainingTimeSeconds = training_times)
```




<table class="data-frame"><thead><tr><th></th><th>Benchmark</th><th>Accuracy</th><th>Loss</th><th>trainingTimeSeconds</th></tr><tr><th></th><th>String</th><th>Any</th><th>Float64</th><th>Any</th></tr></thead><tbody><p>8 rows × 4 columns</p><tr><th>1</th><td>Azure VM 1; CPU; Benchmark 1</td><td>0.880783</td><td>0.4649</td><td>3.54387</td></tr><tr><th>2</th><td>Azure VM 1; GPU; Benchmark 1</td><td>0.876758</td><td>0.480444</td><td>0.255519</td></tr><tr><th>3</th><td>Azure VM 1; CPU; Benchmark 2</td><td>0.925517</td><td>0.267284</td><td>9.0776</td></tr><tr><th>4</th><td>Azure VM 1; GPU; Benchmark 2</td><td>0.927475</td><td>0.258824</td><td>0.512827</td></tr><tr><th>5</th><td>Azure VM 1; CPU; Benchmark 3</td><td>0.962342</td><td>0.131131</td><td>18.0053</td></tr><tr><th>6</th><td>Azure VM 1; GPU; Benchmark 3</td><td>0.960758</td><td>0.141972</td><td>2.46613</td></tr><tr><th>7</th><td>Azure VM 1; CPU; Benchmark 4</td><td>-</td><td>NaN</td><td>-</td></tr><tr><th>8</th><td>Azure VM 1; GPU; Benchmark 4</td><td>0.997875</td><td>0.0156178</td><td>11.1517</td></tr></tbody></table>



As expected, GPU usage offers speed improvement relative to CPU for training the neural network. 

### Next

Azure VM 1; CPU; Benchmark 4 threw an error: ```ERROR: LoadError: Loss is NaN``` - investigate further.
