
"""
ref:https://github.com/ttgump/scDCC/blob/master/layers.py#L23
"""
const lg = loggamma
zinbLoss(Θ,M, Π,X ) = begin
    eps = 1f-7
    # # zinb
    if X < eps
        # the probability of zero is equal to
        # dropout rate Π add the probability that get 0 count from NB distribution
        zero_case =  -log(Π + (1f0-Π)* (Θ/(Θ+M+eps))^Θ + eps )
    else
        # the probability of get X count from the NB distribution
        nb_case =
            # t1
            lg(Θ+eps) + lg(X+1f0) - lg(X+Θ+eps) +
            # t2
            (Θ+X) * log(1f0 + (M/(Θ+eps))) +
            (X * (log(Θ+eps) - log(M+eps))) -
            # t3
            log(1f0-Π+eps)
    end
end

edgeLoss(edgeAdj, X̂, X) = begin
    D = sum(edgeAdj,dims=2) |> vec
    D = diagm(D.^-1.0)
    # if we wanna each cluster have the most same value
    # a = D * edgeAdj * X̂'
    # mse(a,X')
    #
    mean( D * edgeAdj * ((X̂ .- X) .^ 2) )

end

# relation_loss(relationAdj::Matrix, DegreeAdj::Matrix, X̂::Matrix, X::Matrix) =
#     mean( DegreeAdj * relationAdj * ((X̂ .- X) .^ 2) )

labelmatrix(label) = begin
    s = length(label)
    labeladj = zeros(Float32, s, s)
    for i in 1:s, j in 1:s
        if label[i] == label[j]
            labeladj[i,j] = 1.0
        end
    end
    labeladj
end

clusterLoss(labeladj,X̂,X) = begin
    D = sum(labeladj,dims=2) |> vec
    D = diagm(D.^-1.0)
    # a = D * labeladj * X̂'
    # mse(a,X')
    mean( D * labeladj * ((X̂ .- X) .^ 2) )
end


Base.@kwdef mutable struct zinbAEmodel <: Models
    Enc
    Θ
    M
    Π
end
Flux.@functor zinbAEmodel

BuildzinbAE( dim1s::Vector{Int}, dim2s::Vector{Int}; act = softplus) = begin
    layers = Vector{Any}(undef,length(dim1s)-1)
    for i in 1:length(dim1s)-2
        layers[i] = Chain(
            Dense(dim1s[i] => dim1s[i+1]),
            BatchNorm(dim1s[i+1], act)
        )
    end
    layers[end] = Dense(dim1s[end-1] => dim1s[end])
    Enc = Chain(layers...)
    # decorder
    layers = Vector{Any}(undef,length(dim2s)-1)
    for i in 1:length(dim2s)-2
        layers[i] = Chain(
            Dense(dim2s[i] => dim2s[i+1]),
            BatchNorm(dim2s[i+1], act)
        )
    end
    Dec = Chain(layers[1:end-1]...)
    # Θ
    Θ = Dense(dim2s[end-1] => dim2s[end],softplus)
    # M
    M = Dense(dim2s[end-1] => dim2s[end],softplus)
    # Π
    Π = Dense(dim2s[end-1] => dim2s[end],σ)
    zinbAEmodel( Chain(Enc,Dec), Θ, M, Π)
end
BuildzinbAE( dim1s::Vector{Int};kws... ) = BuildzinbAE( dim1s, reverse(dim1s);kws... )

function (m::zinbAEmodel)(X̂)
    enc = m.Enc(X̂)
    m.Θ(enc), m.M(enc),m.Π(enc)
end

"""
AutoEncoder Args.
- `η = (1f-4,1f-6)` the range of learning rate during annealing.
- `λ = 0.01f0` weight decay rate (support in future)
- `batchsize = 128` how many sample will using in one epoch.
- `annealing_cycle = 1` anneal cycle.
- `epochs = 10`  How many epochs will it run, unless it has been converged.
- `seed = 42`
- `use_cuda = false`  whether using the CUDA, GPU is need.
- `checktime = 10`
- `checkplot = false` whether plot the result in each checkpoint.
- `early_stop = 5`
- `tblogger = true` whether record the log.
- `savepath = "log/aemodel/"`
- `denoise = false` whether using the denoise mode.
- `noise_factor = 0.1f0`
- `eval_type = "SIL"` Indicators for evaluation, `SIL`, `ARI`, or `NMI`
- `Enclayers::Vector{Int} = [512,128,64]` the layer and node of encoder
- `Declayers::Vector{Int} = [64,128,512]` the layer and node of decoder
"""
Base.@kwdef struct Args_ae
    η = (1f-3,1f-6)
    # λ = 0.01f0 # weight decay rate
    batchsize = 128
    annealing_cycle = 1
    epochs = 10
    seed = 42
    use_cuda = false
    checktime = 10
    checkplot = false
    early_stop = 5
    tblogger = true
    savepath = "log/aemodel/"
    denoise = false
    noise_factor = 0.1f0
    eval_type = "SIL"
    Enclayers::Vector{Int} = [512,128,64]
    Declayers::Vector{Int} = [64,128,512]
end



"""
根据label 进行 zinb impute
"""
function trainzinbAE end
function trainzinbAE(X::Matrix{Float32},label,adj;
                      model::Union{Models,Nothing}=nothing,
                      filename="zinbAE_res",
                      act = softplus,
                      kws...)
    try
        args = Args_ae(;kws...)
        args.seed > 0 && Random.seed!(args.seed)
        # if args.use_cuda && CUDA.functional()
        #     device = gpu
        #     @info "Training on GPU"
        # else
        #     device = cpu
        #     @info "Training on CPU"
        # end
        curTime = now() |> string
        @info "Start time => $(curTime)"
        savepath = args.savepath*curTime

        adj = Float32.(adj + LinearAlgebra.I)
        S = Float32.(sum(X,dims = 1) / median(sum(X,dims = 1)))
        X̂ = Flux.normalise(@. log(X / S + 1f0))

        # split data into train and validation.
        train_X, val_X = splitdata(X,0.1)
        train_X̂, val_X̂ = splitdata(X̂,0.1)
        train_S, val_S = splitdata(S,0.1)
        train_label, val_label = splitdata(label,0.1)
        train_idx, val_idx = splitdata([1:size(X,2)...],0.1)

        #trainData = (train_X,train_X̂,train_S)
        trainData = Flux.Data.DataLoader(
            (train_X,train_X̂,train_S,train_label,train_idx),
            batchsize=minimum([args.batchsize,size(train_X,2)]))
        valiData = (val_X,val_X̂,val_S,val_label,val_idx)

        # define the model
        if isnothing(model)
            Enclayers = args.Enclayers
            if Enclayers[1] != size(X,1)
                pushfirst!(Enclayers, size(X,1))
            end
            model = BuildzinbAE(Enclayers, act = act)
        else
            @info "using prior model"
        end
        function eval_loss(model, (X, X̂, S, label, idx),adj)
            (Θ,M,Π) = model(X̂)
            M = M .* S
            # # zinb
            loss1 = mean(zinbLoss.(Θ, M, Π, X ))
            labeladj = labelmatrix(label)
            edgeAdj = adj[idx,idx]
            loss2 = sum(edgeLoss(edgeAdj, M, X) ) / size(X,2)
            loss3 = sum(clusterLoss(labeladj,M,X) ) /size(X,2)
            loss1 + loss3 + loss2
        end

        eval_loss1(args...) = begin
            eval_loss(args...) |> x ->  round(x, digits=4)
        end
        function report(epoch)
            val = eval_loss1(model, valiData, adj)
            println("Epoch: $epoch  validated loss: $(val))")
        end

        # using cos annealing technologic
        cos_anneal = cos_anneal_args(args.annealing_cycle,
                                     args.η[1],
                                     args.η[2])

        report(0)
        batchs_in_epoch = length(trainData )
        progress = Progress(args.epochs*batchs_in_epoch )
        ps = Flux.params(model)

        # logging
        if args.tblogger
            tblogger = TBLogger(savepath, tb_overwrite)
            set_step_increment!(tblogger, 0)
            @info "TBlogging at \"$(savepath)\""
        end

        # earlystop
        init_score = eval_loss1(model, valiData, adj)
        es = Flux.early_stopping((x)->x,
                                 args.early_stop;
                                 init_score=init_score)

        # Flux.testmode!(model,:auto)
        for epoch in 1:args.epochs
            batch = 0
            for (X, X̂, S, label, idx) in trainData
                batch += 1
                lr = cos_anneal(batch,epoch,batchs_in_epoch)
                opt = RMSProp(lr)
                labeladj = labelmatrix(label)
                edgeAdj = adj[idx,idx]
                loss, back= Flux.pullback(ps) do
                    (Θ,M,Π) = model(X̂)
                    M = M .* S
                    # # zinb
                    loss1 = mean(zinbLoss.(Θ, M, Π, X ))
                    loss2 = sum(edgeLoss(edgeAdj, M, X) ) / size(X,2)
                    loss3 = sum(clusterLoss(labeladj,M,X) ) /size(X,2)
                    loss1 + loss3 + loss2
                end
                gs = back(1f0)
                Flux.Optimise.update!(opt, ps, gs)
                # loss = eval_loss(model, X, X̂, S)
                # Flux.train!(eval_loss, ps,[(model, X, X̂, S)], opt )
                next!(progress;
                      showvalues=[(:loss, loss),
                                  (:step, (epoch-1)*batchs_in_epoch+ batch ),
                                  (:lr,lr)])
                # logging info about the batch
                set_step!(tblogger, (epoch-1)*batchs_in_epoch + batch )
                with_logger(tblogger) do
                    @info "lr" lr = lr
                    @info "batch" loss = loss
                end

            end
            set_step!(tblogger, epoch* batchs_in_epoch)
            with_logger(tblogger) do
                v_loss = eval_loss1(model, valiData, adj)
                # t_loss = eval_loss1(model, trainData, adj)
                @info "valid" loss = v_loss
            end
            # save model
            if epoch % args.checktime == 0
                @info epoch
                !ispath(savepath) && mkpath(savepath)
                curTime = now() |> string
                modelPath = joinpath(savepath,"$(curTime).bson")
                gae_model = model |> cpu
                let
                    BSON.@save modelPath gae_model args
                end
                @info "[checkpoint] Model saved in \"$(modelPath)\""
                # Flux.testmode!(gcn_model)
                args.checkplot &&
                    begin
                        if isnothing(label)
                            l = LouvainLabel(model.Enc(g,X))
                        else
                            l = label
                        end
                        vis_cluster(model.Enc(g,X),
                                    l,
                                    epoch=epoch,
                                    tblogger=tblogger,
                                    eval_type = args.eval_type,
                                    step=epoch*batchs_in_epoch,
                                    filename = filename)
                    end
            end
            # earlystop
            es( eval_loss1(model, valiData, adj) ) && break
        end
        return model
    catch err
        if typeof(err) == InterruptException
            @error "Interrupted by user, current model is returned"
            return model
        else
            print(err)
        end
    end
end
#+end_src
