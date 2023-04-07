


"""
聚类结果可视化
input a matrix, usine umap to descrease-dim, plot the 2d scatter
other kws, see UMAP.jl
"""
function plotCluster end
plotCluster(X,label;kws...) = plotCluster(X;label=label,kws...)
function plotCluster(X;label=nothing,
                     colorSet=prettycolor6,
                     cmap = :blues,
                     colorvalue = nothing,
                     filename = "filename",
                     size = (800,800),markersize = 10,
                     kws...)
    CairoMakie.activate!()
    # caculated the distance
    ack = [:min_dist,:spread,:n_neighbors,:metric,:local_connectivity]
    kws1 = kwscheck(kws, ack)
    embedding = umap(X,dims=2;kws1...)

    ack = [ :L2norm,:compareOn,:backend,:return_fig ]
    kws2 = kwscheck(kws, ack)
    plotScatter(embedding;label=label,
                colorSet=colorSet,
                cmap=cmap, markersize = markersize,
                filename = filename,
                colorvalue=colorvalue,
                size = size,L2norm = false,kws2...)
end

"""
Cluster result plot
plotCluster(d,:emb2D)
plotCluster(d,:layout2D)
"""
plotCluster(d::scData;on=:emb2D,label = nothing,kws...) = begin
    @assert !isnothing(getproperty(d,on))
    CairoMakie.activate!()
    plotScatter(getproperty(d,on);label = label,kws...)
end
