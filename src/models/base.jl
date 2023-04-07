abstract type Models end

"""
a basic GAE model for intergrate the graph structure and node feature
"""
Base.@kwdef mutable struct GAEmodel <: Models
    Enc
    Dec
end
Flux.@functor GAEmodel

function (m::GAEmodel)(g,X)
    enc = m.Enc(g,X)
    adj = enc |> m.Dec
    (adj = adj, emb = enc)
end

function (m::GAEmodel)(d::scData)
    (adj,emb) = m(d.graph, Float32.(d.emb))
    addMeta!(d, (adjust_adj = sparse(adj .> 0.5), adjust_emb = emb,) )
    @info "result stored in Metainfo as :adj and :emb"
end
function adjust!(d::scData)
    @assert typeof(d.metaInfo.model) <: Models
    d.metaInfo.model(d)
end


function BuildGAE( dim1s::Vector{Int};
                   type = :GAT, BN = true,
                   act = softplus,kws... )

    GCN(ch::Pair,act;kws...) = begin
        ack = [:add_self_loops, :use_edge_weight]
        kws = kwscheck(kws, ack)
        GCNConv(ch[1] => ch[2],act; kws...)
    end

    SAGE(ch::Pair,act;kws...) = begin
        ack = [:aggr, :bias]
        kws = kwscheck(kws, ack)
        SAGEConv(ch[1] => ch[2],act;kws...)
    end

    GAT(ch::Pair,act;kws...) = begin
        ack = [:heads, :concat, :bias, :negative_slope, :add_self_loops]
        kws = kwscheck(kws, ack)
        GATConv(ch[1] => ch[2],act;kws...)
    end

    typeDict = Dict(
        :GCN => GCN,
        :SAGE => SAGE,
        :GAT => GAT,
    )

    Convlayer = typeDict[type]
    act1 = BN ? identity : act
    layers = []
    for i in 1:length(dim1s)-2
        push!(layers,Convlayer(dim1s[i] => dim1s[i+1], act1;kws...))
        BN && push!(layers, BatchNorm(dim1s[i+1], act))
    end
    push!(layers, Dense(dim1s[end-1] => dim1s[end]))

    Enc = GNNChain(layers...)
    Dec = x ->  σ(x' * x)
    GAEmodel(Enc, Dec)
end
