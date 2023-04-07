"""
Graph AutoEncoder Args.
- `η = (1f-3,1f-6)` the range of learning rate during annealing.
- `annealing_cycle = 1` anneal cycle.
- `epochs = 10`  How many epochs will it run, unless it has been converged.
- `seed = 42`
- `use_cuda = false`  whether using the CUDA, GPU is need.
- `checktime = 10`
- `early_stop = 5`
- `tblogger = true` whether record the log.
- `savepath = "log/gaemodel/"`
- `Enclayers::Vector{Int} = [512,128,64]` the layer and node of encoder
- `Declayers::Vector{Int} = [64,128,512]` the layer and node of decoder
"""
Base.@kwdef mutable struct GAETrainArgs
    η = (1f-3, 1f-6) # Float32
    epochs = 10
    seed = 42
    usecuda = false
    annealing_cycle = 1
    early_stop = 5
    tblogger = true
    savepath = "log/gaemodel/"
    Enclayers::Vector{Int} = [512,128]
    # Declayers::Vector{Int} = [128,512]
    checktime = 100
end
"""
Train a GAE model,

- model: 在模型基础上继续训练
- act:: 模型激活函数
- type [:GCN, :SAGE, :GAT]
- debug
"""
function train(d::scData;
                  model::Union{Models,Nothing}=nothing,
                  act = relu,  label=nothing, # for debug
                  debug = false, type = :GCN, # :GCN, :SAGE, :GAT
                  need_normalise = true,skipCompare = true,
                  kws...)
    @assert typeof(d.graph) <: GNNGraph
    g = d.graph
    @assert g.num_nodes == size(d.X,2) == size(d.Y,2)
    # if need_normalise
    # X = d.Y # |> Flux.normalise
    # else
    #     X = d.Y
    # end
    X = Float32.(d.emb)
    try # for capture the Ctrl-c
        ack = kwscheck(kws, fieldnames(GAETrainArgs))
        args = GAETrainArgs(;ack...)
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

        g_mat = Matrix(adjacency_matrix(g) + LinearAlgebra.I)
        g_mat = Float32.(g_mat.>0.2) |> device
        # plot for debug
        if debug
            plotGraph(g,label, filename="inputAdj",markersize=15)
        end

        # get the idx of pos and neg edge.
        n = g.num_edges
        idx = g.graph[3] .> 0.1
        e = edge_index(g)
        fixed_posi = [(i,j) for (i,j) in zip(e[1][idx],e[2][idx] )]
        posidx = CartesianIndex.(fixed_posi)
        negidx = negEdgeSample(g,length( fixed_posi), fixed_posi)
        # split data into train and validation.
        train_pos, test_pos = splitdata(posidx,0.7,seed = 42)
        train_neg, test_neg = splitdata(negidx,0.7, seed = 42)


        trainidx = vcat(train_pos, train_neg)
        testidx = vcat(test_pos, test_neg)
        totalidx = vcat(posidx, negidx)
        g = g |> device
        X = Float32.(X) |> device

        # get the idx for compare.
        compareOn = d.compare_idx
        # setting the loss
        if (!isnothing(compareOn) && length(compareOn) > 0) || !skipCompare
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

        # define the model
        if isnothing(model)
            Enclayers = args.Enclayers
            if Enclayers[1] != size(X,1)
                pushfirst!(Enclayers, size(X,1))
            end
            model = BuildGAE(Enclayers;act = act,type = type,kws...)
        else
            @info "using prior model"
        end
        model = model |> device
        #using cos annealing technologic
        cos_anneal = cos_anneal_args(args.annealing_cycle,
                                     args.η[1],
                                     args.η[2])
        #伪batchs = 10
        batchs_in_epoch  = 10
        progress = ProgressMeter.Progress(args.epochs*batchs_in_epoch)
        ps = Flux.params(model)
        # logging
        if args.tblogger
            tblogger = TBLogger(savepath, tb_overwrite)
            set_step_increment!(tblogger, 0)
            @info "TBlogging at \"$(savepath)\""
        end
        # earlystop
        # init_loss = loss(model, g, X, g_mat,testidx,compareOn) |> x -> round(x, digits=4)
        # @info "Initial loss = $(init_loss)"
        # es = Flux.early_stopping((x)->x,
        #                          args.early_stop;
        #                          init_score=init_loss)
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
            (adj,tmp) = model(g, X)
            aa = adj .> 0.5
            plotGraph(GNNGraph(aa),label, filename="0_Adj")
           plotCluster(tmp,label, filename="0_emb")
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
                    @info "testloss" test = loss(model, g, X, g_mat,testidx,compareOn)
                    @info "trainloss" train = loss(model, g, X, g_mat,trainidx,compareOn)

                    (adj,_) = model(g, X)
                    aa = adj .> 0.5
                    @info "total_acc" indices = confuseDict_(aa, g_mat, totalidx,
                                                       length(posidx),
                                                       length(negidx))
                    # plot for debug
                    if debug
                        plotGraph(GNNGraph(aa),label, filename="$(epoch)_Adj",markersize=15)
                                    (_,tmp) = model(g,X)
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
        model::GAEmodel |> cpu |> x -> addMeta!(d,(model = x,))
    catch err
        if typeof(err) == InterruptException
            @warn "Interrupted by user, current model is returned"
            model::GAEmodel |> cpu |> x -> addMeta!(d,(model = x,))
        else
            print(err)
        end
    end
    @info "Use `adjust!()` function to apply your model"
end

function confuseDict_(Ŷ,Y, idx, length_T, length_N; tostrings=false)
    Ŷ,Y = Ŷ|>cpu,Y|>cpu
    if tostrings
        pretty = x -> "$(round(x*100,digits=3))%"
    else
        pretty = x -> x
    end
    @info sum(Ŷ[idx] .== 1)
    @info sum(Y[idx] .== 1)
    Dict(:TP =>  (sum(Ŷ[idx] .== Y[idx] .== 1) / length_T) |> pretty,
    :TN => (sum(Ŷ[idx] .== Y[idx] .== 0) / length_N)  |> pretty,
    :FP => (sum((Ŷ[idx] .== 1) .* (Y[idx] .== 0)) / length_N) |> pretty,
    :FN => (sum((Ŷ[idx] .== 0) .* (Y[idx] .== 1)) / length_T) |> pretty,
    )
end
