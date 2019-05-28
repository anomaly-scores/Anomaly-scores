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
using FileIO

include("experimentalutils.jl")

data_folder = mainfolder * "results/svae_latent_dim_test/"

function subsample(data, labels, max_anomalies)
    anom_idx = labels .== 1
    norm_data = data[:, .!anom_idx]
    norm_labels = labels[.!anom_idx]

    num_anom_idx = findall(anom_idx)
    new_anom_idx = randperm(length(num_anom_idx))[1:min(max_anomalies, length(num_anom_idx))]

    return hcat(norm_data, data[:, new_anom_idx]), vcat(norm_labels, labels[new_anom_idx])
end

function process_file(f)
    println("$f: loading...")
    param_df = CSV.read(data_folder * f)
    dataset = param_df[:dataset][1]
    i = param_df[:i][1]
    hdim = param_df[:hdim][1]
    ldim = param_df[:ldim][1]
    num_pseudoinputs = param_df[:num_pseudoinputs][1]
    β = param_df[:β][1]
    γ = param_df[:γ][1]
    β_str = β == 1 ? "1.0" : β == 10 ? "10.0" : "$β"
    γ_str = γ == 1 ? "1.0" : "$γ"
    run_name = "$dataset-$i-$hdim-$ldim-$num_pseudoinputs-$β_str-$γ_str"
    if isfile(data_folder * "$run_name-large_metrics_is.csv")
        println("Skipping $f because it was processed already...")
        return
    end
    svae = deserialize(data_folder * "$run_name-svae.jls")
    (x_train, labels_train) = deserialize(data_folder * "$dataset-$i-train.jls")
    (x_test, labels_test) = deserialize(data_folder * "$dataset-$i-test.jls")

    
    println("$f: training discriminator...")
    # println("$f: training discriminator...")
    # d, disc_loss_train = train_disc(svae, x_train)
    # disc_loss_train = Flux.Tracker.data(disc_loss_train)
    # mean_disc_scores_z_train = Tuple(mean(d(zparams(svae, x_train)[1].data), dims = 2).data)
    # mean_disc_scores_z_test = Tuple(mean(d(zparams(svae, x_test)[1].data), dims = 2).data)
    # mean_disc_scores_z_fake = Tuple(mean(d(sampleVamp(svae, size(x_train, 2))), dims = 2).data)
    println("$f: skipping training discriminator...")
    disc_loss_train = -1
    mean_disc_scores_z_train = (-1, -1)
    mean_disc_scores_z_test = (-1, -1)
    mean_disc_scores_z_fake = (-1, -1)

    println("$f: computing metrics...")
    auc_pxv_x_train, auc_pxv_z_train, auc_pz_train, auc_pz_jaco_enco_train, auc_pz_jaco_deco_train, auc_pxv_pz_train, auc_pxv_pz_jaco_enco_train, auc_pxv_pz_jaco_deco_train, log_pxv_x_train, log_pxv_z_train, log_pz_train, log_pz_jaco_enco_train, log_pz_jaco_deco_train, log_px_is_train, auc_px_is_train = compute_metrics(svae, x_train, labels_train .- 1)
    auc_pxv_x_test, auc_pxv_z_test, auc_pz_test, auc_pz_jaco_enco_test, auc_pz_jaco_deco_test, auc_pxv_pz_test, auc_pxv_pz_jaco_enco_test, auc_pxv_pz_jaco_deco_test, log_pxv_x_test, log_pxv_z_test, log_pz_test, log_pz_jaco_enco_test, log_pz_jaco_deco_test, log_px_is_test, auc_px_is_test = compute_metrics(svae, x_test, labels_test .- 1)
    
    df = DataFrame(dataset = dataset, i = i, hdim = hdim, ldim = ldim, num_pseudoinputs = num_pseudoinputs, β = β, γ = γ, auc_pxv_x_train = auc_pxv_x_train, auc_pxv_z_train = auc_pxv_z_train,
                auc_pz_train = auc_pz_train, auc_pz_jaco_enco_train = auc_pz_jaco_enco_train, auc_pz_jaco_deco_train = auc_pz_jaco_deco_train, auc_pxv_pz_train = auc_pxv_pz_train, auc_pxv_pz_jaco_enco_train = auc_pxv_pz_jaco_enco_train,
                auc_pxv_pz_jaco_deco_train = auc_pxv_pz_jaco_deco_train, log_pxv_x_train = log_pxv_x_train, log_pxv_z_train = log_pxv_z_train, auc_pxv_x_test = auc_pxv_x_test, auc_pxv_z_test = auc_pxv_z_test,
                auc_pz_test = auc_pz_test, auc_pz_jaco_enco_test = auc_pz_jaco_enco_test, auc_pz_jaco_deco_test = auc_pz_jaco_deco_test, auc_pxv_pz_test = auc_pxv_pz_test, auc_pxv_pz_jaco_enco_test = auc_pxv_pz_jaco_enco_test,
                auc_pxv_pz_jaco_deco_test = auc_pxv_pz_jaco_deco_test, log_pxv_x_test = log_pxv_x_test, log_pxv_z_test = log_pxv_z_test, disc_loss_train = disc_loss_train, mean_disc_scores_z_train = mean_disc_scores_z_train, mean_disc_scores_z_test = mean_disc_scores_z_test, mean_disc_scores_z_fake = mean_disc_scores_z_fake,
                log_pz_train = log_pz_train, log_pz_jaco_enco_train = log_pz_jaco_enco_train, log_pz_jaco_deco_train = log_pz_jaco_deco_train, log_pz_test = log_pz_test, log_pz_jaco_enco_test = log_pz_jaco_enco_test, log_pz_jaco_deco_test = log_pz_jaco_deco_test)


    df[:log_px_is_train] = log_px_is_train
    df[:log_px_is_test] = log_px_is_test
    df[:auc_px_is_train] = auc_px_is_train
    df[:auc_px_is_test] = auc_px_is_test
                
    CSV.write(data_folder * "$run_name-large_metrics_is.csv", df)
end

function train_disc(svae, x)
    z = zparams(svae, x)[1].data
    d = Chain(Dense(size(z, 1), 64, swish), Dense(64, 64, swish), Dense(64, 2, identity), softmax)
    zfake = sampleVamp(svae, size(z, 2) * 2).data
    data = hcat(z, zfake)
    labels = Flux.onehotbatch(vcat(ones(size(z, 2)), zeros(size(z, 2) * 2)), 0:1)
    loss(x, y) = Flux.crossentropy(d(x), y)
    cb = Flux.throttle(() -> println(loss(data, labels)), 2)
    opt = Flux.Optimise.ADAM()
    Flux.train!(loss, Flux.params(d), RandomBatches((data, labels), 100, 10000), opt, cb = cb)
    return d, loss(data, labels)
end

function compute_metrics(model, x, labels, compute_mmd = false)
    z = zparams(model, x)[1].data
    xp = model.g(z).data
    zp = zparams(model, xp)[1].data

    z_mmd_dst = nothing
    z_mmd_pval = 0
    if compute_mmd
        println("computing null dst...")
        z_norm = z[:, labels .== 0]
        z_inds = randperm(size(z_norm, 2))[1:min(300, size(z_norm, 2))]

        null_dst, null_γ = null_distr_distances(model, length(z_inds))

        println("computing mmd dst...")
        z_mmd_dst = 0
        for i in 1:10
            z_mmd_dst += Flux.Tracker.data(IPMeasures.mmd(IPMeasures.IMQKernel(null_γ), sampleVamp(model, length(z_inds)).data, z_norm[:, z_inds], IPMeasures.pairwisecos))
        end
        z_mmd_dst /= 10
        z_mmd_pval = mmdpval(null_dst, z_mmd_dst)
    end

    println("computing likelihoods...")
    log_pxv_x = vec(collect(FewShotAnomalyDetection.log_normal(x, xp)'))
    log_pxv_z = vec(collect(sum((z .- zp) .^ 2, dims = 1)'))

    log_pz_ = vec(collect(log_pz(model, x)'))
    log_px_is = log_px(model, x)
    log_pz_jaco_enco = vec(collect(log_pz_jacobian_encoder(model, x)'))
    log_pz_jaco_deco = vec(collect(log_pz_jacobian_decoder(model, z)'))

    println("computing aucs...")
    auc_pxv_x = computeauc(.-log_pxv_x, labels)
    auc_pxv_z = computeauc(.-log_pxv_z, labels)
    auc_pz = computeauc(.-log_pz_, labels)
    auc_px_is = computeauc(.-log_px_is, labels)
    auc_pz_jaco_enco = computeauc(.-(log_pz_jaco_enco), labels)
    auc_pz_jaco_deco = computeauc(.-(log_pz_jaco_deco), labels)
    auc_pxv_pz = computeauc(.-(log_pxv_x .+ log_pz_), labels)
    auc_pxv_pz_jaco_enco = computeauc(.-(log_pxv_x .+ log_pz_jaco_enco), labels)
    auc_pxv_pz_jaco_deco = computeauc(.-(log_pxv_x .+ log_pz_jaco_deco), labels)

    if compute_mmd
        return z_mmd_dst, z_mmd_pval, auc_pxv_x, auc_pxv_z, auc_pz, auc_pz_jaco_enco, auc_pz_jaco_deco, auc_pxv_pz, auc_pxv_pz_jaco_enco, auc_pxv_pz_jaco_deco, mean(log_pxv_x), mean(log_pxv_z), mean(log_pz_), mean(log_pz_jaco_enco), mean(log_pz_jaco_deco), mean(log_px_is), auc_px_is 
    else
        return auc_pxv_x, auc_pxv_z, auc_pz, auc_pz_jaco_enco, auc_pz_jaco_deco, auc_pxv_pz, auc_pxv_pz_jaco_enco, auc_pxv_pz_jaco_deco, mean(log_pxv_x), mean(log_pxv_z), mean(log_pz_), mean(log_pz_jaco_enco), mean(log_pz_jaco_deco), mean(log_px_is), auc_px_is
        
    end
end

function null_distr_distances(model, k = 500)
    z1 = sampleVamp(model, 500).data
    z2 = sampleVamp(model, 500).data 
    γ = get_γ(z1, z2)
    null_dst = zeros(Float32, 100)
    for i in 1:100
        z1 = sampleVamp(model, k).data
        z2 = sampleVamp(model, k).data
        null_dst[i] = IPMeasures.mmd(IPMeasures.IMQKernel(γ), z1, z2, IPMeasures.pairwisecos)
    end
    sort!(null_dst)
    return null_dst, γ
end

function get_γ(x, y)
    γs = -10:0.05:2
    cs = [IPMeasures.crit_mmd2_var(IPMeasures.IMQKernel(10.0 ^ γ), x, y, IPMeasures.pairwisecos) for γ in γs]
    γ = 10 ^ γs[argmax(cs)]
end

mmdpval(null_dst, x) = searchsortedfirst(null_dst, x) / length(null_dst)

files = filter(f -> occursin("run_params.csv", f), readdir(data_folder))
dataset = "no dataset"
it = 0
# if length(ARGS) != 0
#     println("ARGS[1] = $(ARGS[1])")
#     dataset = ARGS[1]
#     if length(ARGS) > 1
#         println("ARGS[1] = $(ARGS[1]) ARGS[2] = $(ARGS[2])")
#         it = parse(Int, ARGS[2])
#     end
#     if (it > 0)
#         files = filter(f -> occursin("$dataset-$it-", f), files)
#     else
#         files = filter(f -> occursin(dataset, f), files)
#     end
# end

# if length(ARGS) > 2
files = reverse(files)
# end

for (i, f) in enumerate(files)
    if isfile(data_folder * f)
        println("$dataset-$it: $i/$(length(files))")
        process_file(f)
    end
end
