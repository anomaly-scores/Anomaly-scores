using FewShotAnomalyDetection
using Flux
using MLDataPattern
using FluxExtensions
using Adapt
using DataFrames
using CSV
using Serialization
using Random
using Statistics
using IPMeasures

include("experimentalutils.jl")

outputFolder = mainfolder * "results/svae_goodness_of_fit_dev/"
mkpath(outputFolder)

function runExperiment(datasetName, train, test, inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, num_pseudoinputs, β, γ, batchSize = 100, numBatches = 10000, i = 0)
    X = train[1]
    
    println("Creating SVAE...")
    svae = SVAEvampmeans(size(X, 1), hiddenDim, latentDim, numLayers, nonlinearity, layerType, num_pseudoinputs)
    loss(data) = wloss(svae, data, β, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, γ))
    cb_loss(data) = decomposed_wloss(svae, data, β, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, γ))
    opt = Flux.Optimise.ADAM(1e-4)
    cb = Flux.throttle(() -> println("SVAE RE, Wass. dst: $(cb_loss(X))"), 10)

    println("Started training...")
    Flux.train!(loss, Flux.params(svae), RandomBatches((X,), size = batchSize, count = numBatches), opt, cb = cb)

    serialize(outputFolder * "$datasetName-$i-$hiddenDim-$latentDim-$num_pseudoinputs-$β-$γ-svae.jls", svae)
    df_params = DataFrame(dataset = datasetName, idim = inputDim, hdim = hiddenDim, ldim = latentDim, layers = numLayers, num_pseudoinputs = num_pseudoinputs, β = β, γ = γ, i = i)
    CSV.write(outputFolder * "$datasetName-$i-$hiddenDim-$latentDim-$num_pseudoinputs-$β-$γ-run_params.csv", df_params)
    return df_params
end

batchSize = 100
iterations = 10000

i = 0
dn = "non-existing-default"

if length(ARGS) != 0
    dn = ARGS[1]
    i = parse(Int, ARGS[2])
end
df = "easy"

println("Loading data...")
train, test, clusterdness = loaddata(dn, df)
train = ADatasets.subsampleanomalous(train, 0.1)
println("Saving data...")
serialize(outputFolder * "$dn-$i-train.jls", train)
serialize(outputFolder * "$dn-$i-test.jls", test)

evaluateOneConfig = p -> runExperiment(dn, train, test, size(train[1], 1), p..., batchSize, iterations, i)
println("Started gridsearch...")
gridsearch(evaluateOneConfig, [64 32], [2, 4, 9], [3], ["swish"], ["Dense"], [1, 4, 16], Float32.([0.01, 0.1, 1., 10.]), Float32.([1., 0.1, 0.01, 0.001]))

