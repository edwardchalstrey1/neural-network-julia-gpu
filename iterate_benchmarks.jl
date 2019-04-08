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