
"""
cos annealing strategy usage:
```
cycle = 1 # 1 epoch
lr_max,lr_min = 1f-3,1f-6
f = cos_anneal_args(cycle,lr_max,lr_min)
cur_batch,cur_epoch = 2,3
batchs_a_epoch = 10
lr = f(cur_batch,cur_epoch,batchs_a_epoch)
opt = ADAM(lr)
```
ref: https://arxiv.org/abs/1608.03983
"""
Base.@kwdef mutable struct cos_anneal_args
annealing_cycle = 1
lr_max = 0.001f0
lr_min = 0.000001f0
end
(m::cos_anneal_args)(batch,epoch,batchs_in_epoch) = begin
    Ti = m.annealing_cycle
    Tcur = (epoch - 1) % Ti + batch / batchs_in_epoch
    if batchs_in_epoch == 1 && Ti == 1
        1f-4
    else
        m.lr_min + 1/2*(m.lr_max - m.lr_min)*(1 + cos(  Tcur/Ti*π ))
    end
end

"""
perform a PCA and return top n principal components
args:
- X: the data
- pratio=1: how much percentage of variants to be explained.
- outdim=20: how much dims to output.

the final dims is depended on both pratio and outdim.

```julia
res = PCA(X;outdims=20)

```
ref:
1. https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
1. https://plotly.com/julia/pca-visualization/

How many principal components are needed?
Unfortunately, there is no well-accepted objective way to decide how many principal components are enough (James et al. 2013). In fact, the question depends on the specific area of application and the specific data set. However, there are three simple approaches, which may be of guidance for deciding the number of relevant principal components.

These are

    the visual examination of a scree plot
    the variance explained criteria or
    the Kaiser rule. *

"""
function PCA end
PCA(X;pratio=1,outdim) = begin
    @assert 0 < pratio <= 1
    if isnothing(outdim)
        outdim = size(X,2)
    end
    @assert 1 <= outdim <= size(X,2)
     M = MultivariateStats.fit(MultivariateStats.PCA, X;
                               pratio = pratio, maxoutdim = outdim)
     # caculate the var %
     EV = MultivariateStats.tprincipalvar(M) / MultivariateStats.tvar(M) * 100
     @info "Explaination var: r^2 =  $(round(EV,digits=2)) %"
     MultivariateStats.predict(M, X)
end
"""
Select principal components by var% explained

```julia
res = PCA(X, 0.8)
```

"""
function PCA(X,n::Real)
        @info "max $(n * 100)%"
    M = MultivariateStats.fit(MultivariateStats.PCA, X,
                              pratio = 1)
    pct = M.prinvars ./ M.tvar
    sm =  1/size(X,1)
    cs = cumsum(pct)
    cut = [1:length(pct)...][(cs .> n)][1] #.&& (pct .< sm)][1]
    # return hcat(cs .>n, cs, pct.<sm)
    #Δ = pct[1:end-1] - pct[2:end]
    # cut2  = findfirst( x->x < sm, pct)

    # cut = !isnothing(cut2) ? minimum([cut,cut2]) : cut

    @info "Auto selection, nfeats= $(cut)"
    EV = sum(MultivariateStats.principalvars(M)[1:cut]) / MultivariateStats.tvar(M) * 100
    @info "Explaination var: r^2 = $(round(EV,digits=2)) %"
    MultivariateStats.predict(M,X)[1:cut,:]
end

"Build a KNN graph.
given an matrix, with sample in column, and a optional k (default is 5), \\
return a Knn-Graph::GNNGraph

```julia
KNNGraph(X,k = 5)
```

It is faster than its counterpart in GraphNeuralNetworks.
ref: https://github.com/dillondaudert/UMAP.jl/blob/15dd45e4f40361bb90644fb8df327a74632201b0/src/utils.jl#L111-L122
"
function KNNGraph end
KNNGraph(X,k;kws...) = KNNGraph(X; k = k,kws...)
function KNNGraph(X::AbstractMatrix{S};
                  k = 5,seed = 42,
                  distance = Euclidean(),
                  ignore_diagonal = true,
                  kws...) where {S <: Real}
    k =  size(X,2) > k ? k : size(X,2) - 1
    dist_mat = pairwise(distance,X,dims = 2)
    # find the top k neighbors
    range = (1:k) .+ ignore_diagonal
    knbs = Array{Int,2}(undef, k, size(X,2))
    dists = Array{S,2}(undef,k,size(dist_mat,2))
    for i ∈ 1:size(dist_mat, 2)
        knbs[:,i]  = partialsortperm(dist_mat[ :, i], range, rev = false)
        dists[:,i] = dist_mat[knbs[:,i],i]
    end
    g = GNNGraph([knbs[:,i] for i in 1:size(knbs,2)],
                 gdata = (neighbors = k, nearNN = knbs, dist = dists))
end

"Build a weighted KNN graph
given an matrix, with sample in column, and optional k (default is 5), \\
return a Knn-Graph::GNNGraph
```julia
wKNNGraph(X,k = 5)
```
"
function wKNNGraph end
wKNNGraph(X,k;kws...) = wKNNGraph(X; k = k,kws...)
function wKNNGraph(X::AbstractMatrix{S};
                  k = 5,seed = 42, n::Int = 2,
                  distance = Euclidean(),
                  ignore_diagonal = true,
                  kws...) where {S <: Real}
    k =  size(X,2) > k ? k : size(X,2) - 1
    dist_mat = pairwise(distance,X,dims = 2)
    cor_mat = cor(X)
    # find the top k neighbors
    range = (1:k) .+ ignore_diagonal
    knbs = Array{Int,2}(undef, k, size(X,2))
    dists = Array{S,2}(undef,k,size(dist_mat,2))
    weights = ones(Float32,k,size(dist_mat,2))
    for i ∈ 1:size(dist_mat, 2)
        knbs[:,i] = partialsortperm(dist_mat[ :, i], range, rev = false)
        dists[:,1] = dist_mat[knbs[:,i],i]
        weights[:,i] = cor_mat[knbs[:,i],i]
    end
    # matrix to COO
    res = ones(Float32,size(X,2) * k, 4)
    weights = weights.^n
    for i ∈ 1:size(dist_mat, 2)
        rg = ((i-1)*k+1):i*k
        res[rg,1] .= i
        res[rg,2] = knbs[:,i]
        res[rg,3] = dists[:,i]
        res[rg,4] = weights[:,i]
    end
    g = GNNGraph( Int.(res[:,1]),
                  Int.(res[:,2]),
                  res[:,4],
                  gdata = (neighbors = k,
                           nearNN = knbs,
                           dist = dists,
                           weights = weights))
end


cutKNN2!(d::scData;cutoff=2) = begin
    knbs = d.knns
    Dist = d.dists
    sc,sr = size(Dist,2),size(Dist,1)
    cut_idx = repeat([sr],sc)
    Threads.@threads for i in 1:sc
        for j in 2:sr
            if (Dist[j,i] > cutoff*Dist[j-1,i])
                #cut_idx[i] = j-1
                d.adj[knbs[j,i],i] = d.adj[i,knbs[j,i]] = 0
            end
        end
    end
    d.graph = GNNGraph(d.adj)
end

"""
using share neighbor info to optimize graph. \\
for each pair point, their edge will discard once
their share neigbors is less than `cutoff`.

the

```julia
g = KnnGraph(X,k = 15)
graph_plot(g,label)

g1 = cutKnn(g;cutoff = 1)
g2 = cutKnn(g;cutoff = 2)

graph_plot(g1,label)
graph_plot(g2,label)
```
"""
cutKNN(g;cutoff::Int = 2) = begin
    @assert cutoff >= 1
    Knbs = g.gdata.nearNN
    adj_mat = adjacency_matrix(g) |> Matrix
    n = size(Knbs,2)
    snns = Array{Float64}(undef,n, n)
    Threads.@threads for i in 1:n
        for j in i:n
            snns[i,j] = snns[j,i] = snn_score(Knbs[:,i], Knbs[:,j])
        end
    end
    # cut optimized
    adj_mat[snns .< cutoff/size(Knbs,1)] .= 0.
    GNNGraph(adj_mat)
end

cutKNN1(g;cutoff::Int = 2) = begin
    @assert cutoff >= 1
    Knbs = g.gdata.nearNN
    adj_mat = adjacency_matrix(g) |> Matrix
    n = size(Knbs,2)
    sn = cutoff /size(Knbs,1)
    sc,sr = size(Knbs,2),size(Knbs,1)
    dropMask = zeros(n,n)

    Threads.@threads for i in 1:sc
        for j in 2:sr
            node1 = i
            node2 = Knbs[j,i]
            if snn_score(Knbs[:,node1], Knbs[:,node2]) >= sn
                dropMask[node1,node2] = dropMask[node2,node1] = 1
            end
        end
    end

    GNNGraph(adj_mat .* dropMask,
             gdata = (nearNN = Knbs, dist = g.gdata.dist))
end
snn_score(x,y) = begin
    length(∩(x, y)) / length(∪(x, y))
end




"""
评估聚类效果，
- silhouettes width
另外参考  [`ARI`](@ref) , [`NMI`](@ref)

ref: https://juliastats.org/Clustering.jl/stable/validate.html

ref: Peter J. Rousseeuw (1987). Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis. Computational and Applied Mathematics. 20: 53–65.
"""
function SIL(X,label;precomputed=false,distance=Euclidean())
    label = labelToAssignment(label)
    d = precomputed ? X : pairwise(distance, X, dims=2)
    silhouettes(label, d) |> mean
end

"""
评估聚类效果，
- ARI
另外参考  [`SIL`](@ref) , [`NMI`](@ref)

ref: https://juliastats.org/Clustering.jl/stable/validate.html

ref: Lawrence Hubert and Phipps Arabie (1985). Comparing partitions. Journal of Classification 2 (1): 193–218
"""
function ARI(label,label_true)
    label = labelToAssignment(label)
    label_true = labelToAssignment(label_true)
    randindex(label,label_true)[1]
end

"""
评估聚类效果，
- NMI
另外参考  [`silhouettes width`](@ref) , [`ARI`](@ref)

ref: https://juliastats.org/Clustering.jl/stable/validate.html

ref: Proceedings of the 26th Annual International Conference on Machine Learning - ICML ‘09.
"""
function NMI(label,label_true)
    label = labelToAssignment(label)
    label_true = labelToAssignment(label_true)
    mutualinfo(label,label_true; normed=true)
end

ld = pyimport("leidenalg")
ig = pyimport("igraph")
"""leiden by python"""
function pyleiden(g;resolution = 0.5)
    ld = pyimport("leidenalg")
    ig = pyimport("igraph")
    Random.seed!(42)
    # julia graph to igraph
    # generate the edge list
    el = hcat(g.graph[1],g.graph[2])
    el .-= 1 # python index from 0
    # create nework from edgelist
    G = ig.Graph(edges=el)
    partition = ld.find_partition(G,
                           ld.CPMVertexPartition,
                           resolution_parameter = resolution)
    label = partition._membership .+ 1
end

"""K-means by scikit-learn"""
function pykmeans(X; k = 5)
    sklearn = pyimport("sklearn.cluster")
    kmeans = sklearn.KMeans
    label = kmeans(n_clusters=k, random_state=0).fit(X')
    label.labels_ .+ 1
end

"""
split data into train and validation.
return (train dataset, validation dataset)
"""
splitdata(X::Array,η=0.7;seed = 42) = begin
    @assert  0.0 < η < 1.0

    seed > 0 && Random.seed!(seed)
    c = size(X)[end]
    s = floor(c * η,digits=0) |> Int
    idx = shuffle(1:c)
    # (X[:,s+1:end],X[:,1:s])
    (selectdim(X, length(size(X)), 1:s),
     selectdim(X, length(size(X)), s+1:c))
end


# distance compare
# L1Distance(X̂ , X ) = map(StatsBase.L1dist, eachcol(X̂), eachcol(X))

L1Distance(x::SubArray,y::SubArray) = mean(abs.(x .- y))
L1Distance(X̂::Matrix, X::Matrix) = mean(map(L1Distance, eachcol(X̂), eachcol(X)))

cosineSim(X̂::SubArray,X::SubArray) = (X̂' * X) / (√(sum(X̂.^2)) *  √(sum(X.^2)))
cosineSim(X̂::Matrix,X::Matrix) = mean(map(cosineSim, eachcol(X̂), eachcol(X)))

RMSE(X̂::SubArray,X::SubArray) =  sum( @. (X̂ - X)^2) / length(X)
RMSE(X̂::Matrix,X::Matrix) = mean(map(RMSE, eachcol(X̂), eachcol(X)))


"""
a small function used to filte the kws,

```
kws = (a=1,b=2,c=3,)
ack = [:a] # only accept the kws "a"
kws = kwscheck(kws,ack)
# then pass to the function
f(;kws...) # only the keyword arg "a" will be passed.
```
"""
# kws map and dispatch
kwscheck(k::pairs(NamedTuple),ack) = begin
    kws = intersect(keys(k),ack)
    k[kws]
end


function umap(X;dims=2,seed = 42,
              min_dist::Real = 0.1, # 0.3 in uwot
              spread::Real = 1,
              n_neighbors::Int = 15,
              metric = Euclidean(), # or cosine
              learning_rate::Real = 1,
              local_connectivity::Integer = 1
              )
    Random.seed!(seed)
    UMAP.umap(X, dims,
              min_dist = min_dist, # for match with uwot
              spread = spread,
              n_neighbors = n_neighbors,
              metric = metric,
              learning_rate = learning_rate,
              local_connectivity = local_connectivity)
end



function leidenLabel(A;resolution = 0.8, randomness = 0.01, metric = Euclidean)
    res = Leiden.leiden(A,
                        resolution = resolution,
                        randomness = randomness).partition
    # res to label
    label = ones(Int,size(A,2))
    for i in 1:length(res)
       label[res[i]] .= i
    end
    label
end

"""
kmeansLabel(X;k = 5,metric = Euclidean)
"""
function kmeansLabel(X;k = 5,metric = Euclidean)
    kmeans( X, k) |> assignments
end

"""
从节点标签中抽取部分节点, 返回index

分类等分抽样
```julia
randomlabel(labelList; cut::Int = 2,seed = 42)
```
- cut::Int  按类抽取抽取的比例，eg: cut = 10, 则每一类抽取 1/10.

指定比例/数量抽样
```julia
randomlabel(labelList, n::Real = 2;seed = 42)
randomlabel(labelList, n::Real = 0.2;seed = 42)
```
- n::Int  抽取的总数
- n::Float  抽取的比例
"""
function randomlabel end
function randomlabel(labelList; cut::Int = 2,seed = 42)
    Random.seed!(seed)
    types = unique(labelList)
    idxs = [findall(==(i),labelList) for i in types]
    vcat([sample(i, length(i) ÷ cut ;replace=false) for i in idxs]...)
end
function randomlabel(labelList,n::Real;seed = 42)
    Random.seed!(seed)
    if typeof(n) == Int
        sample(1:length(labelList), n)
    else
        sample(1:length(labelList), length(labelList) * n |> Int ∘ round )
    end
end
"""
return the discrete vector
res  = discrete(X)
"""
discrete(X::Array) = begin
    Float32.(X .>  median(X;dims=2))
end


"convent label to numeric assignment vector"
function labelToAssignment(label)
    if any( x -> !isa(x,Int), label )
        a_ = unique(label)
        label_num = Dict()
        for i in 1:length(a_)
            label_num[a_[i]] = i
        end
        label = [label_num[i] for i in label]
    end
    label
end


### retired function

"""
hclust
"""
hclustLabel(X;k = 5,metric = Euclidean) = begin
    d = pairwise(Euclidean(), X, dims=2)
    res = hclust(d)
    cutree(res,k = k)
end
