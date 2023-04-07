module MSGNN
using Reexport
using MKL
using Random
@reexport using SparseArrays,DataFrames, CSV
@reexport using StatsBase,Dates
@reexport using Flux, BSON, CUDA, PyCall, RCall
@reexport using CairoMakie, NetworkLayout
@reexport using LinearAlgebra, SpecialFunctions
@reexport using GraphNeuralNetworks
@reexport using ColorSchemes

import MultipleTesting,MultivariateStats,JLD2,UMAP

using HypothesisTests: MannWhitneyUTest, pvalue
using Flux: onecold, onehotbatch
using Flux.Losses: logitbinarycrossentropy, mse, kldivergence
using ProgressMeter,Logging
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!

using Distances, Clustering
import Term: Panel, highlight_syntax
import Term.Dendograms: Dendogram


########
# api  #
########

include("dataprocess/dataStruct.jl")
export setData,
    addFeature!,
    delFeature!,
    addMeta!,
    delMeta!,
    checkName

include("dataprocess/preProcess.jl")
export libraryNormalized,
    relateScale,
    relateScale!,
    cluster!,PCA!,umap!,leiden!,
    buildGraph!

include("dataprocess/priorKnowledge2.jl")
export compareOn!

include("dataprocess/utils.jl")
export cos_anneal_args,KNNGraph,SIL,NMI,ARI,splitdata,kwscheck,
    labelToAssignment,splitdata

include("models/base.jl")
export adjust!

include("models/pairwiseDistance.jl")
include("models/train.jl")
include("models/train_dev.jl")
include("models/loss.jl")
export train,train2
# dev model
# include("models/SIGNmodel.jl")
# include("models/ZINBImpute.jl")
# include("models/train_dev.jl")

##############
#  可视化相关 #
##############
include("Visual/prettycolors.jl")
include("Visual/base.jl")
include("Visual/scatterplot.jl")
include("Visual/clusterplot.jl")
include("Visual/graphplot.jl")
export
    plotCluster,
    plotScatter,
    plotGraph,
    graphLayout!,
    linePair,
    lineHvsL,
    Pair_idx,
    HLidx
    # plotRange


############
# 分析相关  #
############
# include("downAnalysis/aucell_dev.jl")
include("downAnalysis/clusterCompare.jl")
include("downAnalysis/DE.jl")
export dist, dist_2,dist_compare,
    heatmap, findMarkers, findAllMarkers
include("downAnalysis/enrich.jl")
export enrichGO, compareGO

################
# 其他工具比较  #
################
include("downAnalysis/seurat.jl")
export seurat

include("downAnalysis/interaction.jl")
export cellSelect


help() = begin
    dendo = Dendogram("Main",
                      "setData",
                      "cluster",
                      "train",
                      "adjust",
                      "analysis")
    setData =  """
1. setData
2. addFeature (optional)
3. compareOn (optional)
4. cluster
"""

    currentmodel = """
1. GCN (default)
2. GAT
3. SAGE
"""

    analysis = """
1. plotCluster
2. plotGraph
3. heatmap
4. enrich
5. findmarker
6. cellSelect
"""

    print(Panel(dendo, subtitle="steps",width = 75, style="green", justify = :left),
          Panel(setData, subtitle="setData",width = 25, justify = :left) *
              Panel(currentmodel, subtitle="model",width = 25, justify = :left) *
              Panel(analysis, subtitle="analysis",width = 25, justify = :left))
end
export help

# import Term.Dendograms: link

# mydend = Dendogram("cluster", [filter, buildGraph, , 2])
# otherdend = Dendogram("other", [:a, :b])

# print(
#     link(mydend, link(otherdend, otherdend; title="another level"); title="One level")
# )
end
