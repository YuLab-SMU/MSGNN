########################
# preprocess functions #
########################

"""
以样本总count数的中位数作为对齐的参考，其他样本向其缩放对齐，
使得每个样本的总counts约等于中位，然后再进行对数转换。

```julia
X = libraryScale(X)
```
参考自 DCA 预处理
"""
function libraryNormalized(X)
     S = Float32.(sum(X,dims = 1) / median(sum(X,dims = 1)));
     log1p.(@. X / S)
end

"""
对样本数据进行缩放(scale), 使得每个样本的总reads counts相等,
实际上，等同于把counts换成相对丰度，再进行缩放和对数转换，
scalefactor 起到放大样本间距离的作用。默认1e4，参考 Seurat。

`x / total(x) * scalefactor`

```julia
X = relateScale(X,scalefactor=1e4)
```

"""
function relateScale(X,scalefactor=1e4)
    m = minimum(X)
    X .= X .- m
    log1p.(X ./ sum(X,dims = 1) .* scalefactor)
end

function relateScale!(d::scData,kws...)
    d.normdata = relateScale(d.rawdata,kws...)
    return nothing
end

"""
filter HGV

保留差异最大的Top N feature
```julia
filter!(d::scData; top = 3000)
```
"""
function Base.filter!(d::scData; top = 3000, kws...)
    if isnothing(d.normdata)
        @warn "Data has not been normalised, do it with default args."
        relateScale!(d)
    end
    x = d.normdata
    idx = sortperm(std(x,dims=2)|>vec,rev=true)[1:top]
    d.X = x[idx,:]
    d.Xname = d.rawfeature[idx]
    return nothing
end

"""
downsample of the data and label(optional)

"""

function downSample(X::Matrix,
                    label::Union{Vector,Nothing}=nothing;
                    n = 3000,
                    seed=42,kws...)
    s = size(X,1)
    if !isnothing(label)
       @assert size(X,1) == length(label) "label doesnot match data dims"
    end
    @assert n < s && size(X,1) "Args n is larger than data samples"
    Random.seed!(seed)

    if s > n
        idx = rand(1:s,n)
        X = X[idx,idx]
        label =  !isnothing(label) ? label[idx] : label
    end

    if !isnothing(label)
        return (X,label)
    else
        return X
    end
end



#######################
# api relate function #
#######################


"""
warp the function for single cell cluster
steps:
X -> featureFileter -> PCA -> umap -> graphAdj -> cluster

pca args
- out_dim = 30,
- pca_pratio = 0.9,
umap args
- min_dist::Real = 0.3, # for match with uwot
- spread::Real = 0.1,
- n_neighbors::Int = 15,
- metric = Euclidean(), # or cosine
- learning_rate::Real = 1,
- local_connectivity::Integer = 1
cluster args
- resolution = 0.001
"""
function cluster!(d::scData;seed = 42,
                  top::Int = 0,
                  cutoff=2,
                  pca_dim = :auto,
                  pca_pratio = 0.9,
                  resolution = 0.001,
                  relatescale = false,
                  kws...)
    Random.seed!(seed)
    top = size(d.rawdata,1) >= top > 0 ? top : size(d.rawdata,1)
    if relatescale
        relateScale!(d)
    else
        d.normdata = d.rawdata
    end

    filter!(d, top = top)
    PCA!(d; pca_dim=pca_dim,pca_pratio=pca_pratio)
    buildGraph!(d;kws...)
    # cutKNN2!(d;cutoff=cutoff)
    leiden!(d;resolution = resolution)
end


# """
# steps that can be also perform seperately.

# d -> filter!
# """
# function Base.filter!(d::scData;Xfeatures::Int,kws...)
#     idx = sortperm(std(d.X,dims=2)|>vec,rev=true)
#     d.X_filted = idx[1:Xfeatures];
# end
# function filter! end

"""
pca args
- pca_dim::Int,
- pca_pratio
-
"""
function PCA!(d::scData;pca_dim,pca_pratio,kws...)
    if isnothing(d.X)
        @info "running filter! with default args. or you can do it manually"
        Base.filter!(d;kws...)
    end
    @views if pca_dim == :auto # cutoff = pca_pratio
        d.pc = PCA(d.X,pca_pratio)
    else
        d.pc = PCA(d.X,outdim=pca_dim)
    end
    return nothing
end


"""
umap args
- min_dist::Real = 0.3, # for match with uwot
- spread::Real = 0.1,
- n_neighbors::Int = 15,
- metric = Euclidean(), # or cosine
- learning_rate::Real = 1,
- local_connectivity::Integer = 1


d -> filter! -> PCA! -> umap!
"""
function umap!(d::scData;kws...)
    if isnothing(d.pc)
       @info "running PCA! with default args. or you can run it manually"
       PCA!(d;kws...)
    end
    ack = [:min_dist,:spread,:n_neighbors,:metric,:local_connectivity]
    kws = kwscheck(kws, ack)
    M = UMAP.UMAP_(d.pc,2;kws...)
    d.adj =  Float32.(M.graph) # sparse matrix
    # d.Adj[d.Adj .> 0] .= 1f0 #  save weight？ no
    d.emb2D = M.embedding
    d.dists = M.dists
    d.knns = M.knns
    d.graph = GNNGraph( d.adj )  # float32 for better speed in Flux
    return nothing
end


buildGraph!(d::scData;kws...) = begin
    if isnothing(d.pc)
        @info "running PCA! with default args. or you can run it manually"
        PCA!(d;kws...)
    end
    ack = [:min_dist,:spread,:n_neighbors,:metric,:local_connectivity]
    kws = kwscheck(kws, ack)
    # build graph from adj1
    M1 =  UMAP.UMAP_(d.pc,2;kws...)

    # for other feature Y
    if !isnothing(d.Y)
        if  size(d.Y,1) > 2
            M2 =  UMAP.UMAP_(d.Y,2;kws...)
            idx = (M1.graph .> 0.2) .* (M2.graph .> 0.2)
            addMeta!(d,(debugidx = idx,))
            addMeta!(d,(M1 = M1,))
            addMeta!(d,(M2 = M2,))
            d.emb = vcat(M1.embedding, M2.embedding)
            d.emb2D = MSGNN.PCA(d.emb,outdim=2)
            # d.emb2D = UMAP.UMAP_(d.emb,2).embedding

            d.adj =  UMAP.UMAP_(d.emb,2 ).graph
            d.adj[d.adj .> 0.5] .= 0.5
            d.adj[idx] .= 1
        else
            M2 = Flux.normalise(d.Y)
            idx = (M1.graph .> 0.2) .* (M2 .> 0)[1,:]
            # d.emb2D = UMAP.UMAP_(vcat(M1.embedding, M2.embedding),2;kws...).embedding
            d.emb2D = M1.embedding .+ M2
            d.emb = vcat(M1.embedding, M2)
            d.adj =  UMAP.UMAP_(d.emb,2 ).graph
            d.adj[idx] .= 1
        end
    else
        d.emb =  M1.embedding
        d.emb2D = M1.embedding
        d.adj =  M1.graph
    end
    d.adj[diagind(d.adj)] .= 1
    d.graph = GNNGraph( d.adj )
    d.adj = Float32.(d.adj)# float32 for better speed in Flux
    return nothing
end


"""
d -> filter! -> PCA! -> umap! -> leiden!
"""
function leiden!(d::scData;kws...)
    if isnothing(d.graph)
        @info "running umap! with default args. or you can run it manually"
        umap!(d;kws...)
    end
     Random.seed!(42)
    ack = [:resolution]
    kws = kwscheck(kws, ack)
    d.label = pyleiden(d.graph;kws...)
    @info "label num : $(unique(d.label) |> length)"
    return nothing
end

buildGraph!(mtx::Matrix) = begin
   GNNGraph(UMAP.UMAP_(mtx,2 ).graph)
end


"""
prepare for GAE training.
"""
buildGraph1!(d::scData;kws...) = begin
    if isnothing(d.adj)
        @info "running umap! with default args. or you can run it manually"
        umap!(d;kws...)
    end
    d.graph = GNNGraph(d.adj)
end



##########################
# some helpful functions #
##########################

# to be retired.
"""
compareAdj

return two adj, one is the top vs last adj,
while 1 represent that they are in defference clusters,
the other one is the same cluster adj, 1 represent they lie in same cluster.
"""
priorAdj(X,compareOn::Int;kws...) =
    priorAdj(X,[compareOn];kws...)
priorAdj(X,compareOns::Vector{Int};TopN=10) = begin
    s = size(X,2)
    @assert s > 2 * TopN
    adj1 = BitMatrix(zeros(s,s))
    adj2 = BitMatrix(zeros(s,s))
    for n in compareOns
    adj1 =  adj1 .| (topAdj(X,n,TopN) .== 2)
    adj2 =  adj2 .| (topAdj(X,n,TopN) .== 1) .| (topAdj(X,n,TopN) .== 4)
    end
    # less >> more
    (Int.(adj1), Int.(adj2))
end

topAdj(X,compareOn::Int,TopN) = begin
    idx = sortperm(X[compareOn,:])|>vec
    code = repeat([0],size(X,2))
    code[idx[1:TopN]] .= 1
    code[idx[end-TopN+1:end]] .= 2
    code * code'
end



"""
方差小于cutoff 的特征将被丢弃
```julia
lowVarFilter!(d::scData;on::Symbol = :X,cutoff = 0.0)
```
"""
function lowVarFilter!(d::scData;on::Symbol = :X,cutoff = 0.0)
    X = getproperty(d,on)
    idx = std(d.X,dims=2) .> cutoff
    setproperty!(d, on, X[vec(idx),:])
    setproperty!(d, :geneName, d.geneName[ vec(idx)])
    d
end


"""
去掉高dropout的样本，droput rate > 0.8 或者 只剩下<10个非空样本的特征将被丢弃。
```julia
highDropFilter!(d::scData;on = :X, rate = 0.8,rareNum = 10)
```
"""
function highDropFilter!(d::scData;on::Symbol = :X, rate = 0.8,rareNum = 10)
    X = getproperty(d,on)
    idx1 = vec( (sum(X .< 1.0,dims= 2) ./ size(X,2)) .< rate)
    idx2 = vec( sum(cor(X') .> 0.5,dims = 2) .> rareNum )
    setproperty!(d, on, X[ idx1 .| idx2,:])
    setproperty!(d, :geneName, d.geneName[ idx1 .| idx2])
    d
end
