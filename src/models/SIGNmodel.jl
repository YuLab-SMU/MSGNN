Base.@kwdef mutable struct SIGNmodel <: Models
    preCompute
    Ws
    MLP
end
Flux.@functor SIGNmodel
Flux.trainable(d::SIGNmodel) =  (d.Ws, d.MLP)


(m::SIGNmodel)() = begin
   vcat((m.preCompute .* m.Ws)...) |> MLP
end


function NormalizeDegree(adj)
    # 1. adj + I
    A = adj + I
    # 2. caculate the Degree of A
    D = vec(sum(A,dims=2))
    # 3. D⁻½
    D_inv_sqrt = inv.( sqrt.(D) )
    replace!(D_inv_sqrt,Inf=>0,-Inf=>0)
    D = Diagonal(D_inv_sqrt)
    sparse( D * A * D)
end


function BuildSIGN(X, adj, GCNoutdim::Int, k::Int,
                   MLPlayers::Vector{Int};
                   act = softplus,
                   self_loops = true)
    if self_loops
        adj = adj + LinearAlgebra.I
    end

    NLapA = NormalizeDegree(adj)

    preCompute = []
    k = length(GCNkernel)
    for i in 1:k
        push!(preCompute, X^i * NLapA )
    end

    Random.seed!(42)

    weights = []
    for i in 1:k
        push!(weights, Flux.glorot_uniform(GCNoutdim, size(X,1)) )
    end

    GCN = vcat(preCompute .* weights)

    layers = []
    for i in 1:length(MLPlayers)-2
        push!(layers, Dense(MLPlayers[i] => MLPlayers[i+1]))
        push!(layers, BatchNorm(MLPlayers[i+1], act))
    end
    push!(layers, Dense(MLPlayers[end-1] => MLPlayers[end]))

    SIGNmodel(GCN, chain(layers...))
end

Base.@kwdef mutable struct args_SIGN
    η = (1f-3, 1f-6) # Float32
    epochs = 10
    seed = 42
    usecuda = false
    annealing_cycle = 1
    early_stop = 5
    tblogger = true
    savepath = "log/gaemodel/"
    k::Int = 3
    GCNoutdim::Int = 16
    MLPlayers::Vector{Int} = [128,512]
    checktime = 10
    checkplot = false
end

@inline function eval_loss(model,Y)
    mse(model(), Y)
end

trainSIGN(d::scData;
    model::Union{Models,Nothing}=nothing,
    act = softplus,
    kws...) = begin
        @assert typeof(d.graph) <: GNNGraph
        G = d.graph
        @assert G.num_nodes == size(d.X,2) == size(d.Y,2)
        X = d.Y

        try
        args = args_gae(;kws...)
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

        g = G |> device
        g_mat = Float32.(adjacency_matrix(g) + LinearAlgebra.I)|> Matrix |> device
        X = Float32.(X) |> device

        # split data into train and validation.
        # train_ind, val_ind = splitdata(1:size(X,2),0.1)

        # define the model
        if isnothing(model)
            MLPlayers = args.MLPlayers
            if MLPlayers[end] != args.k * args.GCNoutdim
                pushfirst!(MLPlayers, args.k * args.GCNoutdim )
            end
            model = BuildSIGN(X, g_mat,
                              args.GCNoutdim,
                              MLPlayers,
                              act = softplus,
                              self_loops = true)
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
        init_loss = eval_loss() |> x -> round(x, digits=4)
        @info "Initial loss = $(init_loss)"
        es = Flux.early_stopping((x)->x,
                                 args.early_stop;
                                 init_score=init_loss)
        # Flux.testmode!(model,:auto)
        for epoch in 1:args.epochs
            for batch in 1:batchs_in_epoch
                lr = cos_anneal(batch,epoch,batchs_in_epoch)
                opt = ADAM( lr ) # or RMSProp


                Flux.train!(eval_loss, ps, [()], opt)
                # loss = eval_loss(model, g, X, g_mat)
                next!(progress; showvalues=[(:step, (epoch-1)*batchs_in_epoch + batch)])
                # next!(progress; showvalues=[(:loss, loss)])
                                            # (:step, (epoch-1)*batchs_in_epoch + batch ),
                                            # (:lr,lr)])
                # batch logging
                if args.tblogger
                    set_step!(tblogger, (epoch-1)*batchs_in_epoch + batch )
                    args.tblogger && with_logger(tblogger) do
                        @info "lr" lr = lr
                    end
                end
            end

            # epochs logging
            if args.tblogger
                set_step!(tblogger, epoch* batchs_in_epoch)
                with_logger(tblogger) do
                    @info "loss" total = eval_loss()
                end
            end
            # save model # too big
            # if epoch % args.checktime == 0
            #     @info epoch
            #     !ispath(savepath) && mkpath(savepath)
            #     curTime = now() |> string
            #     modelPath = joinpath(savepath,"$(curTime).bson")
            #     gae_model = model |> cpu
            #     let
            #         BSON.@save modelPath gae_model args
            #     end
            #     @info "[checkpoint] Model saved in \"$(modelPath)\""
            #     # Flux.testmode!(gcn_model)
            # end

            # check earlystop status
            es( eval_loss() |> x -> round(x,digits=4)) && break
        end
        return model::SIGNmodel |> cpu

    catch err
        if typeof(err) == InterruptException
            @error "Interrupted by user, current model is returned"
            return model::SIGNmodel |>cpu
        else
            print(err)
        end
    end
end
