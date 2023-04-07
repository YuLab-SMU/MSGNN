module A
using GraphNeuralNetworks
# for multiOmic extend, a n dim Array should be supported.
Base.@kwdef mutable struct scData
    rawdata::Array
    rawfeature::Union{Nothing, Vector} = nothing
    rawID::Union{Nothing, Vector} = nothing
    normdata::Union{Nothing, Array} = nothing

    label::Union{Nothing, Vector} = nothing # label for node
    metaInfo::NamedTuple = (;) # a flexible storage for other info

    #####################
    # prior knowledge 1 #
    #####################
    Y::Union{Matrix{Float32},Nothing} = nothing # the node feature
    Yname::Union{Vector,Nothing} = nothing

    #####################
    # prior knowledge 2 #
    #####################
    compareOn = nothing # which feature will be compare on
    compare_idx = nothing # the compare feature idx
    _compare_mode = "hvsh" #default, or "hvsl"

    ####################
    # processed record #
    ####################
    X::Union{Array,Nothing} = nothing # the filtered and normalized data
    Xname::Union{Vector,Nothing} = nothing

    emb::Union{AbstractMatrix,Nothing} = nothing
    pc::Union{AbstractMatrix,Nothing} = nothing
    graph::Union{GNNGraph,Nothing}= nothing # the graph adj Matrix
    emb2D::Union{AbstractMatrix,Nothing} = nothing # the Embedding of UMAP
    adj::Union{AbstractMatrix,Nothing}= nothing # the graph adj Matrix
    knns::Union{AbstractMatrix,Nothing} = nothing
    dists::Union{AbstractMatrix,Nothing} = nothing
    layout2d = [nothing,nothing] # the layout of graph
end
end
scData = A.scData

# 访问器
function Base.propertynames(x::scData)
    return ( :X,:Xname,:label,:metaInfo,
             :PCA,:graph,:emb2D,
             :layout2d,:Y,
             :Yname)
end

fun_docs1 = """
functions to add/del property into data struct of scData
- addFeature!
- delFeature!
- addMeta!
- delMeta!
"""

"""
$(fun_docs1)

To add a new feature to the scData.feature, beacause node feature is organized as a 2D Array,
the new feature must provided as n * num_node matrix. new feature will be normalise and
hcat into current feature Array.

```julia
d = setData(rand(10,10))
newFeature = rand(3,10) # assume there are only 5 nodes
featureName = "marker" .* string.([1,2,3])
addFeature!(d, newFeature, featureName )
```
"""
function addFeature! end
function addFeature!(d::scData,
                     featureValue::Vector,
                     featureName::String;
                     replace = false)
    addFeature!(d, reshape(featureValue,1,:), [featureName], replace = replace)
end
function addFeature!(d::scData, featureValue::Matrix,
                     featureName::Vector;replace = false)
    if (d.Y == nothing) | replace
        d.Yname = featureName
        d.Y =  featureValue
    else
        @assert size(featureValue,2) == size(d.Y,2)
        # todo: check the repeated name!!!
        toupdate = intersect(featureName, d.Yname)
        delFeature_!(d, toupdate)
        d.Yname = vcat(d.Yname, featureName)
        d.Y=  vcat(d.Y,featureValue);
    end
end

"""
$(fun_docs1)

To remove node feature(s):

```julia
featureName = "marker" .* string.([1,2,3])
delFeature!(d, featureName )
```
"""
delFeature_!(d::scData, featureName::Vector) = begin
    a = findall(x->x in featureName, d.Yname )
    filter!(e->e ∉ featureName,d.Yname)
    d.Y = d.Y[setdiff(1:end, a), :]
    return nothing
end
delFeature!(d::scData, featureName::Vector) = begin
    delFeature_!(d, featureName )
    if size(d.Y,1) == 0
        d.Y = nothing
        d.Yname = nothing
    end
    return nothing
end
"""
$(fun_docs1)

To add new metadata(s) record for node or feature,

```julia
newRecord = (a = "new record",) # a named tuple
addMeta!(d, newRecord)
```
"""
function addMeta! end
addMeta!(d::scData, meta::NamedTuple) = begin
    d.metaInfo = (;d.metaInfo..., meta...);
    keys(d.metaInfo)
end
"""
$(fun_docs1)
To remove a metadata record for node or feature,
```julia
k = :recordName
delMeta!(d, k)
```
"""
delMeta!(d::scData, k::Symbol) = begin
    @info "drop $(k) from the meta info"
    d.metaInfo = Base.structdiff(d.metaInfo, NamedTuple{(k,)})
    return keys(d.metaInfo)
end
"""
Function to init a scData struct
example
```julia
rawdata = rand(10,10)
d = setData(rawdata)
# with more control
d = setData(rawdata,
rawfeature = "gene" .* string.([1:10...]),
rawID = string.([1:10...])
)
```
"""

setData(rawdata;kws...) = begin
    tmp = scData(rawdata=rawdata)
    (r,c) = size(tmp.rawdata)
    K = keys(kws)
    V = values(kws)
    if :rawfeature in K
        @assert size(V[:rawfeature])[1] == r
        tmp.rawfeature = V[:rawfeature]
    else
        tmp.rawfeature = string.([1:r...])
    end
    for i in [:label, :Y, :rawID]
        if i in K
            @assert size(V[i])[end] == c
            setproperty!(tmp, i, V[i])
        end
    end
    # the value that need to check if eq to feature nums
    #fo r i in [:rawfeature]
    #     if i in K
    #         @assert size(V[i],1) == r
    #         setproperty!(tmp, i, V[i])
    #     end
    # end
    if tmp.Y !== nothing
        tmp.Yname = :Yname ∈ K ?
            V[:Yname] :
            "feature" .* string.(1:size(tmp.Y,1))
    else
        tmp.Yname = nothing
    end
    tmp
end



"""
a function for fuzzy feature name checking
"""
checkName(d::scData, n::String) = begin
    d.Xname[occursin.([n],d.Xname)]
end
checkName(m::Vector, n::String) = begin
    m[occursin.([n],m)]
end

"""
function for friendly display of scData obj
"""
function Base.show(io::IO, d::scData)
    info = ""

    info *= "🔢 raw data size: $(size(d.rawdata))\n"
    promt = isnothing(d.normdata) || isnothing(d.X) ?
        "Data have not been " :
        ""
    if isnothing(d.normdata)
        promt = promt *  "1. relateScale"
    end
    if isnothing(d.X)
        promt = promt *  " 2. feature select"
    end

    # println(promt)
    info *= "\n" * promt * "\n"
    if !isnothing(d.X)
        info *= "🔢 filted X size: $(size(d.X))\n"
    end

    if !isnothing(d.Y)
        info *= "🏷️  Prior feature preview:"
        k = minimum([5,length(d.Yname)])
        for i in 1:k
            info *=  "\n" * string.(d.Yname[i])
        end
    else
        info *= "🏷️  No Prior features added"
    end
    info *= "\n"
    # show label
    if !isnothing(d.label)
        l = unique(d.label) |> sort
        info *= "unique label: $(length(l))\n"
    end
   info *= "\n"
    # show compareOn
    if !isnothing(d.compareOn)
        c = length(d.compareOn)
         info *= "🔗️ compare Pair:\n"
        if c <= 6
            for i in 1:c
                 info *=  string.(d.compareOn[i]) * "\n"
            end
        else
            for i in 1:3
                 info *=  string.(d.compareOn[i]) * "\n"
            end
             info *= "... $(c - 6) rows omited ...\n"
            for i in c-3+1:c
                 info *=  string.(d.compareOn[i]) * "\n"
            end
        end
else
         info *= "🔗️ No compare Pairs added \n"
    end
     info *= "\n"

    if length(keys(d.metaInfo)) > 0
         info *=  "🏁 meta info: $(keys(d.metaInfo))\n"
    else
        info *=  "🏁 No meta info added \n"
    end
 info *= "\n"
    if !isnothing(d.graph)
        # info *= "🕸️  $(d.graph)"
        info *= "🕸️ Graph size: $(d.graph.num_nodes) node, $(d.graph.num_edges) edges"
    else
        info *= "🕸️  no Graph built"
    end
    print(Panel(info))
end
