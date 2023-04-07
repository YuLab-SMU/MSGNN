

"""
find marker gene for label1 vs label2
return both pos and neg marker featurer idx
"""
function findMarkers(expr, lp::Pair, label;
                     fc = 2,
                     qv = 0.05, kws...)
    @assert length(label) == size(expr,2)
   (label1, label2) = lp
    if typeof(label1) <: Vector
        label1 = label1
    else
        label1 = [label1]
    end
    if typeof(label2) <: Vector
        label2 = label2
    else
        label2 = [label2]
    end
    ix1 = indexin(label1, label)
    ix2 = indexin(label2, label)
    res = DEtest(expr[:,ix1], expr[:,ix2])
    sig_pos =  (res[:,6] .< qv) .| (res[:,4] .> fc )
    sig_neg =  (res[:,6] .< qv) .| (res[:,4] .< -1/fc )
    (Int.(res[sig_pos,1]), Int.(res[sig_neg,1]))
end

"""
if only provide one label info,
then find marker gene for label1 vs all other label
return both pos and neg marker featurer idx
"""
function findMarkers(expr, label1, label; kws...)
   @assert length(label) == size(expr,2)

    if typeof(label1) <: Vector
        label1 = label1
    else
        label1 = [label1]
    end

    label2 = setdiff(unique(label), label1 )
    findMarkers(expr, label1 => label2, label;kws...)
end
"""
if provide idx to compare, ,
"""
function findMarkers(expr, idx1::BitVector, idx2::BitVector;
                     fc = 2,
                     qv = 0.05, kws...)
    @assert length(idx1) == length(idx2) == size(expr,2)
    res = DEtest(expr[:,idx1], expr[:,idx2])
    sig_pos =  (res[:,6] .< qv) .* (res[:,4] .> fc )
    sig_neg =  (res[:,6] .< qv) .* (res[:,4] .< -1/fc )
    (Int.(res[sig_pos,1]), Int.(res[sig_neg,1]))
end


"""
find marker gene for each label  vs all other label
return only pos   marker featurer idx
as a dict
"""
function findAllMarkers(expr,label;kws...)

    @assert length(label) == size(expr,2)
    lbuiq = label|>unique

    l = length(lbuiq)
    res = Dict()

    for i in 1:l
        push!(res, "$(lbuiq[i])" => findMarkers(expr, lbuiq[i],label;kws...)[1] )
    end

    res

end

# HypothesisTests
# MultipleTesting
# The Mann-Whitney U test is sometimes known as the Wilcoxon rank-sum test.
function DEtest(X::Matrix,Y::Matrix)
    @assert size(X,1) == size(Y,1)
    res = similar(X, size(X,1), 6) # idx, mean1, mean2, fc, pvalue, fdr
    Eps = eps(res[1,1])
    @views res[:,1] = 1:size(X,1)
    @views res[:,2] = mean(X,dims=2)
    @views res[:,3] = mean(Y,dims=2)
    @views res[:,4] = @. res[:,2] - res[:,3] # cause the data is log2 translate in d.X
    @views for i in 1:size(res,1)
        # a bug, need to update, works only for float64
        res[i,5] = MannWhitneyUTest(Float64.(X[i,:]), Y[i,:]) |> pvalue
    end
    @views res[:,6] = MultipleTesting.adjust(MultipleTesting.PValues(res[:,5]),
                                             MultipleTesting.BenjaminiHochberg())

    res = sortslices(res,dims=1,by=x->(-x[6],x[4]),rev=true)
    res
end
