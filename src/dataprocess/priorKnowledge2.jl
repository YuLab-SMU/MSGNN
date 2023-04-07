###############################################################
# Functions for adding the Pairwise info to guide the cluster #
###############################################################

"""
Functions for adding the prior info to guide the cluster
two kinds of comparision is support
- cells with high A exprs vs high B exprs  should keep aways
- cells with high A exprs should keep aways from those with low A exprs

```julia
# compare between the high-low features
compareOn!(d,["feature1","feature2"])
# compare between the A and B pair
compareOn!(d,["feature1" => "feature2", "feature3" => "feature4"])
```
"""
function compareOn! end

###############################
# if features vector provided  #
###############################
compareOn!(d::scData,on::Int;kws...) =
    compareOn!(d,[on];kws...)
function compareOn!(d::scData,on::Vector{Int};
                    topN::Int=10,
                    replace::Bool = true)
    (X, fts) = _querydata!(d,replace,topN)
    showname = fts[on]

    append!(d.compareOn,
            "(high) " .* string.(showname) .=>  string.(showname) .* " (low)")
    for i in on
        idx = sortperm(X[i,:])|>vec
        # high => low
        push!(d.compare_idx, idx[end-topN+1:end] => idx[1:topN])
    end
    unique!(d.compareOn)
    unique!(d.compare_idx)
    d._compare_mode = "hvsl"
    @info "Comparision: $(d.compareOn[1]) ..."
end

compareOn!(d::scData,on::Vector{String};topN = 10,kws...) = begin
    # check if gene in featureName
    FN = Int[] # store the feture idx
    (X, fts) = isnothing(d.X) ? (d.rawdata, d.rawfeature) :
        (d.X, d.Xname)
    for G in on
        a = findfirst(x->x==G, fts )
        if a != nothing
            push!(FN,a)
        else
            @info "drop $(G), because no conrrespond feature"
        end
    end
    compareOn!(d::scData,FN;topN=topN,kws...)
end


#############################
# if feature pairs provided #
#############################
compareOn!(d::scData,on::Pair;kws...) =
    compareOn!(d,[on];kws...)
function compareOn!(d::scData,
                    on::Vector{Pair{Int,Int}};
                    topN::Int=10,
                    replace::Bool = true)
    (X, fts) = _querydata!(d,replace,topN)
    for P in on
        idx1 = (sortperm(X[P[1],:])|>vec)[end-topN+1:end]
        idx2 = (sortperm(X[P[2],:])|>vec)[end-topN+1:end]
        a1 = setdiff(idx1, idx2)
        a2 = setdiff(idx2, idx1)
        if length(a1) > 0 && length(a2) >0
            push!(d.compare_idx, a1 => a2)
            push!(d.compareOn, fts[P[1]] => fts[P[2]] )
        else
            @info "drop Pair $(P)"
        end
    end
    unique!(d.compareOn)
    unique!(d.compare_idx)
    d._compare_mode = "hvsh"
    @info "Comparision: $(d.compareOn[1]) ..."
end
compareOn!(d::scData,on::Vector{Pair{String,String}};topN=10,kws...) = begin
    # turn string to conrespond number idx
    pair = Pair{Int,Int}[] #Vector{Pair{Int, Int}}[]
    (X, fts) = isnothing(d.X) ? (d.rawdata, d.rawfeature) :
        (d.X, d.Xname)

    for P in on
        a = findfirst(x->x==P[1], fts )
        b = findfirst(x->x==P[2], fts )
        if a != nothing && b != nothing
            push!(pair,a => b)
        else
            @info "drop $(P[1] => P[2]), because no conrrespond feature"
        end
    end
    compareOn!(d, pair;topN=topN,kws...)
end


"""
query,check and return the data
"""
_querydata!(d::scData,replace::Bool,topN::Int) = begin
    if isnothing(d.X)
        if isnothing(d.normdata)
            (X, fts) = (d.rawdata, d.rawfeature)
        else
            (X, fts) = (d.normdata, d.rawfeature)
        end
    else
        (X, fts) = (d.X, d.Xname)
    end

    if isnothing(d.compare_idx) | replace
        d.compare_idx = []
        d.compareOn = []
    end
    @assert size(X,2) > 2 * topN "Arg `topN` should not larger than $(Int(round(size(X,2)/2)))"
    return (X,fts)
end
