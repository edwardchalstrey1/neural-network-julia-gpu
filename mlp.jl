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