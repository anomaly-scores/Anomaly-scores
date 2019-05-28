using ADatasets
using EvalCurves

"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridsearch(f, parameters...) = map(p -> f(p), Base.product(parameters...))

# function printandrun(f, p)
#     println(p)
#     (p, f(p))
# end

global mainfolder = "D:/dev/julia/"
if !isdir(mainfolder)
    mainfolder = "/home/bimjan/dev/julia/" 
end
if !isdir(mainfolder)
    mainfolder = "/app/" 
end
if !isdir(mainfolder)
    error("The main folder is unknown")
end

# Here I will say

const datafolder = mainfolder * "data/loda/public/datasets/numerical"
const server_main_folder = "/home/bimjan/dev/julia/"

loaddata(dataset, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(dataset, difficulty, datafolder)..., 0.8, "low")

computeauc(score, labels) = EvalCurves.auc(EvalCurves.roccurve(score, labels)...)

copytoserver(from, to) = run(`scp $from bayes:$to`)
function create_server_folder(folder) 
    inner_cmd = "mkdir -p $folder"
    run(`ssh bayes $inner_cmd`)
end