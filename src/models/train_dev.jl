"""
a basical model for intergrate the graph structure and node feature
"""
Base.@kwdef mutable struct GAEmodel2 <: Models
    Enc
    Dec = identity
    Xlayer = identity
    Glayer
end
Flux.@functor GAEmodel2

Base.@kwdef mutable struct GAETrainArgs2
    η = (1f-3, 1f-6) # Float32
    epochs = 10
    seed = 42
    usecuda = false
    annealing_cycle = 1
    early_stop = 5
    tblogger = true
    savepath = "log/gaemodel/"
    Enclayers::Vector{Int} = [512,128]
    Declayers::Vector{Int} = [128,512]
    checktime = 10
end

function (m::GAEmodel2)(g,X)
    enc = m.Enc(g,X)
    # (adj = m.Glayer(enc), emb = enc)
    dec = m.Dec(enc)
    (adj = m.Glayer(dec), emb = enc, X̂ = m.Xlayer(dec))
end

function (m::GAEmodel2)(d::scData)
    (adj,emb) = m(d.graph, Float32.(d.emb))
    addMeta!(d, (adjust_adj = sparse(adj .> 0.5), adjust_emb = emb,) )
    @info "result stored in Metainfo as :adj and :emb"
end

"""
build a GAEmodel base on the layers nums provided
"""
function BuildGAE2( dim1s::Vector{Int},
                    dim2s::Vector{Int};
                    type = :GAT, BN = true,
                    act = softplus,
                    droprate = 0.2,kws... )

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
        # BN && push!(layers, Dropout(droprate))
        BN && push!(layers, BatchNorm(dim1s[i+1], act))
    end
    push!(layers, Dense(dim1s[end-1] => dim1s[end]))
    Enc = GNNChain(layers...)

    layers = []
    for i in 1:length(dim2s)-2
        push!(layers, Dense(dim2s[i] => dim2s[i+1], act1))
        BN &&  push!(layers, BatchNorm(dim2s[i+1], act))
    end
    Dec = Chain(layers...)
    Xlayer = Dense(dim2s[end-1] => dim2s[end])

    Glayer = x ->  σ(x' * x)

    GAEmodel2(Enc, Dec, Xlayer, Glayer)
end



BuildGAE2( dim1s::Vector{Int};kws... ) = BuildGAE2( dim1s, reverse(dim1s) ;kws... )



"""
Train a GAE model, 目的是获得有效代表graph的embedding.
使重构出来的 X̂ 和 feature 非常相似， 同时， adĵ 和 adj 也非常相似。
```
m = trainGAE(d::scData)
# or more control
m = trainGAE(d;
    Enclayers = [512,128,64],
    Declayers = [64,128,512],
    η = (1f-3, 1f-6),
    epochs = 10,
    checktime = 5)
```
- model: 在模型基础上继续训练
- act:: 模型激活函数
other kws see [`args_gae`](@ref)
"""
function train2(d::scData;
                  model::Union{Models,Nothing}=nothing,
                  act = relu,  label=nothing, # for debug
                  debug = false, type = :GCN, # :GCN, :SAGE, :GAT
                  need_normalise = true,
                  kws...)
    @assert typeof(d.graph) <: GNNGraph
    g = d.graph
    @assert g.num_nodes == size(d.X,2) == size(d.Y,2)
    X = Float32.(d.emb)
    try # for capture the Ctrl-c
        ack = kwscheck(kws, fieldnames(GAETrainArgs2))
        args = GAETrainArgs2(;ack...)
        args.seed > 0 && Random.seed!(args.seed)
        if args.usecuda && CUDA.functional()
            device = gpu
            @info "Training on GPU"
        else
            device = cpu
            @info "Training on CPU"
        end
        curTime = now() |> string
        @info "Start time => $(curTime)"
        savepath = args.savepath*curTime

        # g_mat = Matrix(adjacency_matrix(g) + LinearAlgebra.I)
          g_mat =  adjacency_matrix(g)
        # g_mat = Float32.(g_mat .> 0) |> device

        # plot for debug
        if debug
            plotGraph(g,label, filename="inputAdj",markersize=15)
            plotGraph(GNNGraph(g_mat),label, filename="inputAdj",markersize=15)
        end
        # get the idx of pos and neg edge.
        #
        #binarycrossentropy(g_mat[idx], adj[idx ])
        #
        n = g.num_edges
        idx = g.graph[3] .> 0.8
        e = edge_index(g)
        fixed_posi = [(i,j) for (i,j) in zip(e[1][idx],e[2][idx] )]
         # fixed_posi = [(i,j) for (i,j) in zip(e[1],e[2])]
        posidx = CartesianIndex.(fixed_posi)
        negidx = negEdgeSample(g,length( fixed_posi), fixed_posi)
        # split data into train and validation.
        train_pos, test_pos = splitdata(posidx,0.7,seed = 42)
        train_neg, test_neg = splitdata(negidx,0.7, seed = 42)

        trainidx = vcat(train_pos, train_neg)
        testidx = vcat(train_neg, test_neg)
        totalidx = vcat(posidx, negidx)

        g = g |> device
        X = Float32.(X) |> device

        # get the idx for compare.
        compareOn = d.compare_idx
        # setting the loss
        if !isnothing(compareOn) && length(compareOn) > 0
            if d._compare_mode == "hvsh"
                @info "we will compare with mode H vs H pair"
                loss = GAELoss_pair
            else
                @info "we will compare with mode H vs L in same featrue"
                loss = GAELoss_high_low
            end
        else
            @info "no compare add"
            loss(model, g, X, g_mat,idx,args...) = GAELoss(model, g, X, g_mat,idx)
        end

        # split data into train and validation.
        # train_ind, val_ind = splitdata(1:size(X,2),0.1)
        # define the model
        if isnothing(model)
            Enclayers = args.Enclayers
            if Enclayers[1] != size(X,1)
                pushfirst!(Enclayers, size(X,1))
            end
            Declayers = args.Declayers
            if Declayers[end] != size(X,1)
                push!(Declayers, size(X,1))
            end
            model = BuildGAE2(Enclayers,Declayers;act = act,type = type,kws...)
        else
            @info "using prior model"
        end

        model = model |> device
        # using cos annealing technologic
        cos_anneal = cos_anneal_args(args.annealing_cycle,
                                     args.η[1],
                                     args.η[2])
        # 伪batchs = 10
        batchs_in_epoch  = 10
        progress = Progress(args.epochs*batchs_in_epoch )
        ps = Flux.params(model)
        # logging
        if args.tblogger
            tblogger = TBLogger(savepath, tb_overwrite)
            set_step_increment!(tblogger, 0)
            @info "TBlogging at \"$(savepath)\""
        end

        # earlystop
        function TNrate(model, g, X, Y, idx, length_T, length_N)
            Ŷ = model(g, X)[1] .> 0.5
            Ŷ,Y = Ŷ|>cpu,Y|>cpu
            tn = sum(Ŷ[idx] .== Y[idx] .== 0) / length_N
            round(tn, digits=4)
        end

        # init_loss = loss(model, g, X, g_mat,testidx,compareOn) |> x -> round(x, digits=4)
        init_loss = TNrate(model,g,X,g_mat,totalidx, length(posidx), length(negidx))
        @info "Initial loss = $(init_loss)"
        es = Flux.early_stopping((x)->x,
                                 args.early_stop;
                                 distance = (best_score, score) -> score - best_score,
                                 init_score=init_loss)

        # Flux.testmode!(model,:auto)
        # plot for debug
        if debug
            (adj,_) = model(g, X)
            aa = adj .> 0.5
            plotGraph(GNNGraph(aa),label, filename="0_")
        end
        for epoch in 1:args.epochs
            for batch in 1:batchs_in_epoch
                lr = cos_anneal(batch,epoch,batchs_in_epoch)
                opt = ADAM( lr ) # or RMSProp

                Flux.train!(loss, ps, [(model, g, X, g_mat,trainidx,compareOn)], opt)

                next!(progress; showvalues=[(:step, (epoch-1)*batchs_in_epoch + batch)])
                # batch logging
                if args.tblogger
                    set_step!(tblogger, (epoch-1)*batchs_in_epoch + batch )
                    args.tblogger && with_logger(tblogger) do
                        @info "lr" lr = lr
                    end
                end
            end

            if device == gpu
                CUDA.reclaim()
            else
                GC.gc()
            end

            # epochs logging
            if args.tblogger
                set_step!(tblogger, epoch* batchs_in_epoch)
                with_logger(tblogger) do
                    @info "test_loss" loss = loss(model, g, X, g_mat,testidx,compareOn)
                    (adj,_,tmp) = model(g, X)
                    aa = adj .> 0.5
                    @info "total_acc" indices = confuseDict_(aa, g_mat, totalidx,
                                                       length(posidx),
                                                       length(negidx))

                    # plot for debug
                    if debug && epoch % 10 == 0
                        plotGraph(GNNGraph(aa),label, filename="$(epoch)_")
                        plotCluster(tmp,label, filename="$(epoch)_emb")

                    end
                end
            end
            # save model
            if epoch % args.checktime == 0
                @info epoch
                !ispath(savepath) && mkpath(savepath)
                curTime = now() |> string
                modelPath = joinpath(savepath,"$(curTime).bson")
                let
                    gae_model = model |> cpu
                    BSON.@save modelPath gae_model args
                end
                @info "[checkpoint] Model saved in \"$(modelPath)\""
                # Flux.testmode!(gcn_model)
            end

            # check earlystop status
            es( loss(model, g, X, g_mat,testidx,compareOn) |> x -> round(x,digits=4)) && break
        end

        (adj,_) = model(g, X)
        aa = adj .> 0.5
        for i in confuseDict_(aa, g_mat, totalidx,
                              length(posidx),
                              length(negidx),
                              tostrings=true)
            println(i)
        end
        model::GAEmodel2 |> cpu |> x -> addMeta!(d,(model = x,))
    catch err
        if typeof(err) == InterruptException
            @warn "Interrupted by user, current model is returned"
             model::GAEmodel2 |> cpu |> x -> addMeta!(d,(model = x,))
        else
            print(err)
        end
    end
     @info "Use `adjust!()` function to apply your model"
end
