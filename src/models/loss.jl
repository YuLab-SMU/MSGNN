# common

function confuseDict_(Ŷ,Y, idx, length_T, length_N; tostrings=false)
    Ŷ,Y = Ŷ|>cpu,Y|>cpu
    if tostrings
        pretty = x -> "$(round(x*100,digits=3))%"
    else
        pretty = x -> x
    end

    Dict(:TP =>  (sum(Ŷ[idx] .== Y[idx] .== 1) / length_T) |> pretty,
    :TN => (sum(Ŷ[idx] .== Y[idx] .== 0) / length_N)  |> pretty,
    :FP => (sum((Ŷ[idx] .== 1) .* (Y[idx] .== 0)) / length_N) |> pretty,
    :FN => (sum((Ŷ[idx] .== 0) .* (Y[idx] .== 1)) / length_T) |> pretty,
    )
    # [ "   " "Pred_T" "Pred_N"; "Real_T" "$(TP*100)%" "$(FN*100)%";"Real_N" "$(FP*100)%" "$(TN*100)%" ]
end


function posEdgeSample(X::Union{G,S},n::Int) where {G<:GNNGraph,S<:SparseMatrixCSC}
    (r,c) = typeof(X)<:GNNGraph ? edge_index(X) : findnz(X)[1:2]
    k = length(r)
    @assert 0 < n
    Random.seed!(42)
    ix = rand(1:k,n)
    CartesianIndex.(r[ix],c[ix])
end

function negEdgeSample(X::Union{G,S},n::Int,fixed_posi) where {G<:GNNGraph,S<:SparseMatrixCSC}
    (r,c) = typeof(X)<:GNNGraph ? edge_index(X) : findnz(X)[1:2]

    k = typeof(X)<:GNNGraph ? X.num_nodes : size(X,2)

    # 固定不能抽取的位置
    fixed_posi = [(i,j) for (i,j) in zip(r,c)]
    append!(fixed_posi,[(i,i) for i in 1:k])
    @assert 0 < n
    # 抽取，直到完全和 posi 相等
    Random.seed!(42)

    t = [(rand(1:k,1)[1],rand(1:k,1)[1]) for i in 1:n]

    setdiff!(t,fixed_posi)
    if length(t) < n # 再抽取一次
        for _ in 1:5
            t2 = [(rand(1:k,1)[1],rand(1:k,1)[1]) for i in 1:n]
            setdiff!(t2,fixed_posi)
            append!(t,t2)
            if length(t) >= n
                break
            end
        end
    end
    t = length(t) > n ? t[1:n] : t
    CartesianIndex.(t)
end


"""
loss is caculated with the pos_edge and neg_edge
"""
@inline function GAELoss(model::GAEmodel,
                         g::GNNGraph,
                         X::AbstractMatrix,
                         g_mat::AbstractMatrix,
                         idx::Vector{CartesianIndex{2}})
    (adj,_) = model(g, X)
    # Flux.Losses.binarycrossentropy(g_mat, adj)
     Flux.Losses.binarycrossentropy(g_mat[idx], adj[idx ])
end

"""
loss is caculated with the pos_edge and neg_edge
"""
@inline function GAELoss(model::GAEmodel2,
                         g::GNNGraph,
                         X::AbstractMatrix,
                         g_mat::AbstractMatrix,
                         idx::Vector{CartesianIndex{2}})
    (adj,_,X̂) = model(g, X)
    mse(X̂,X) + Flux.Losses.binarycrossentropy(g_mat[idx], adj[idx ])
end

"""
Use Student’s t-distribution to measure the pairwise similarities in the given
ref: Learning Embedding Space for Clustering From Deep Representations
"""
function pairwiseSimilar_p(X::AbstractMatrix{Float32};α=1f0)
    # to be fixed, why it is faster in cpu
    a1 = pairwise(Euclidean(),X|>cpu)
    # b1 = sum(a1,dims=1)
    mask = ones(Float32,size(X,2), size(X,2))
    mask -= LinearAlgebra.I
    sparse(mask .* (1f0 .+ a1 .^ 2f0 / α) .^ (-(α + 1f0)/2f0) ./
    (sum( (1f0 .+ a1 ./ α ) .^ (-(α + 1f0)/2f0), dims=1 )))  #|> x -> Float32.(x)
end


function GAELoss_high_low(model::GAEmodel, g, X, g_mat, idx, h_l_idxs ;α=1f0)
    (adj,emb,X̂) = model(g, X)
    n = length(h_l_idxs)
    ls1 = 0f0
    for (idx1,idx2) in h_l_idxs
        l,c = length(idx1), length(idx2)
        O = ones(Float32, l,l )
        o = zeros(Float32, c,c)
        ls1 += Flux.Losses.binarycrossentropy( σ(emb[:,idx1]' * emb[:,idx1]), O )
        ls1 += Flux.Losses.binarycrossentropy( σ(emb[:,idx1]' * emb[:,idx2]), o )
    end
    println(ls1)
    gpu( mse(X̂,X) + ls1 +  Flux.Losses.binarycrossentropy(g_mat[idx], adj[idx ]) )

    # gpu(-10log(pair) - 10log(1f0 - Dist) +
    # Flux.Losses.binarycrossentropy(g_mat[idx], adj[idx ]) )
end

function GAELoss_pair(model::GAEmodel, g, X, g_mat, idx, pairs ;α=1f0)
    (adj,emb, X̂) = model(g, X)
    pair, Dist = 0f0,0f0
    @views for (idx1,idx2) in pairs
        l = length(idx1)
        newdf = pairwiseSimilar_p(emb[:,vcat(idx1,idx2)] , α=α )
        CUDA.@allowscalar pair += mean(newdf[1:l,1:l]) + mean(newdf[l+1:end,l+1:end])
        CUDA.@allowscalar Dist += mean(newdf[1:l,l+1:end]) + mean(newdf[l+1:end,1:l])
    end
    gpu( mse(X̂,X) - log(pair)  -log(1f0 - Dist) +
        Flux.Losses.binarycrossentropy(g_mat[idx], adj[idx ]) )
end
