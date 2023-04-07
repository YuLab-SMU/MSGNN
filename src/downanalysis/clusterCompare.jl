# 聚类距离， 基于 emb 计算
# 量化展示信息整合前后，不同cluster 的距离，解读距离
# eg: 加入cd4 cd8 信息后，相比之前 cd4 高分组细胞距离更近， 说明整合有效
# 然后计算 cluster 两两间距离
# 找出哪些细胞距离在整合信息后发生了明显的改变， 这种改变能提示什么

"""
抽样计算 两类间的某种属性距离 feq:30%
D(r,s) = Trs / ( Nr * Ns)  层次聚类
Complete-linkage clustering - Wikipedia
比如：
"""
function dist(emb,label;sampleRate = 0.3)
   # sample n from each cluster
   k_ = label |> unique
    n = length(k_)
    @assert n > 1
    mtx = ones(n,n)
    for i in 1:n
        idx1 = label .== k_[i]
        for j in 1:n
            idx2 = label .== k_[j]
            if i == j
                mtx[i,j]= 0
            else
                # mtx[i,j]= mean(pairwise(Euclidean(),emb[:,idx1],emb[:,idx2]))
                a = emb[:,idx1]
                b = emb[:,idx2]
                a_1, b_1 = size(a,1), size(b,1)
                a_2 = round(size(a,2) * sampleRate,digits=0) |> Int
                b_2 = round(size(b,2) * sampleRate,digits=0) |> Int
                a = rand(a, (a_1, a_2))
                b = rand(b, (b_1, b_2))
                mtx[i,j]= mean(pairwise(Euclidean(),a,b ))
            end
        end
    end
    mtx ./ maximum(mtx)
end

# 不抽样
function dist_2(emb,label)
   # sample n from each cluster
   k_ = label |> unique
    n = length(k_)
    @assert n > 1
    mtx = ones(n,n)
    for i in 1:n
        idx1 = label .== k_[i]
        for j in 1:n
            idx2 = label .== k_[j]
            if i == j
                mtx[i,j]= 0
            else
                a = emb[:,idx1]
                b = emb[:,idx2]
                a_1, b_1 = size(a,2), size(b,2)
                # mtx[i,j]= mean(pairwise(Euclidean(),emb[:,idx1],emb[:,idx2]))
                mtx[i,j]= sum(pairwise(Euclidean(),a,b)) /  a_1 / b_1
            end
        end
    end
    mtx ./ maximum(mtx)
end

# """
# 差异 找出两个类间距离是否足够大 从而提示存在差异

# Ward’s Method: This method does not directly define a measure of
# istance between two points or clusters.
# It is an ANOVA based approach.
# One-way univariate ANOVAs are done for each variable
# with groups defined by the clusters at that stage of the process.
# At each stage, two clusters merge that provide the smallest increase in
# the combined error sum of squares
# https://online.stat.psu.edu/stat505/lesson/14/14.4
# """


# 热图展示类与类间距离

# function heatmap(mtx;filename="heatmap")
#     Random.seed!(42)
#     fig = Figure(resolution = (600, 400))
#     ax = Axis(fig[1, 1]; xlabel = "cluster", ylabel = "cluster")
#     hmap = heatmap!(mtx; colormap = :Spectral_11)
#     Colorbar(fig[1, 2], hmap; label = "values", width = 15, ticksize = 15, tickalign = 1)
#     colsize!(fig.layout, 1, Aspect(1, 1.0))
#     colgap!(fig.layout, 7)
#     save("$(filename)_$(now()).png", fig)
# end
"""
l = unique(label)
mtx = dist(emb)
heatmap(mtx,l)
"""
function heatmap(mtx;label=nothing,
                 tri = false,text=false,
                 colormap=:bwr, #:seismic
                 filename="heatmap",
                 return_obj = false,
                 resolution = (1200, 600))
    (n,m) = size(mtx)
    if tri
        mtx = copy(mtx)
        for i in 1:n, j in i+1:m
            mtx[i,j] = NaN
        end
    end
    Random.seed!(42)
    if isnothing(label)
        xticks = string.(1:m)
        yticks = string.(n:-1:1)
    elseif typeof(label) <: Tuple
        (xticks,yticks) = label
    elseif typeof(label) <: Vector
        xticks = yticks = label
    end
    xs = repeat([1:m...],outer = n)
    ys = repeat([1:n...],inner = m)
    mtx = mtx[end:-1:1,:]
    zs = mtx' |>vec
    CairoMakie.activate!()
    set_theme!(mytheme)
    fig = Figure(resolution = resolution, fontsize = 20)
    ax = Axis(fig[1, 1], xticks = (1:m, xticks), yticks = (1:n, yticks))
    # hmap = heatmap!(ax, mtx, colormap = :plasma)
    hmap = heatmap!(ax, xs,ys,zs, colormap = colormap)
    # 画图是左到右，下到上，position idx 是先纵再横
    if text
        for i in 1:n, j in 1:m
            txtcolor = mtx[i, j] < 0.2 ? :white : :black
            text!(ax, "$(round(mtx[i,j], digits = 2))", position = (j, i),
                  color = txtcolor, align = (:center, :center))
        end
    end
    Colorbar(fig[1, 2], hmap; label = "values", width = 15, ticksize = 15)
    ax.xticklabelrotation = π / 3
    ax.xticklabelalign = (:right, :center)


# add split line
    # hlines(ys; xmin = 0.0, xmax = 1.0, attrs...)

    hidedecorations!(ax; label = false,
                     ticklabels = false,
                     ticks = false,
                     grid = true,
                     minorgrid = true,
                     minorticks = true)
    save("$(filename)_$(now()).png", fig)
    if return_obj
        fig
    end
end


"""
heatmap(mtx1,mtx2)

"""
function heatmap(mtx,mtx2;label=nothing,save_obj = true, filename="heatmap")
    @assert size(mtx) == size(mtx2)
    (n,m) = size(mtx)
    mtx = copy(mtx)
    for i in 1:m, j in i+1:n
        mtx[i,j] = mtx2[i,j]
    end
    Random.seed!(42)

    if typeof(label) <: Nothing
        xticks = string.(1:m)
        yticks = string.(1:n)
    elseif typeof(label) <: Tuple
        (xticks,yticks) = label
    elseif typeof(label) <: Vector
        xticks = yticks = label
    end

    xs = repeat([1:m...],outer = n)
    ys = repeat([1:n...],inner = m)
    mtx = mtx[end:-1:1,:]
    zs = mtx' |>vec

    CairoMakie.activate!()
    set_theme!(mytheme)
    fig = Figure(resolution = (1200, 600), fontsize = 20)
    ax = Axis(fig[1, 1], xticks = (1:m, xticks), yticks = (1:n, xticks))
    # hmap = heatmap!(ax, mtx, colormap = :plasma)
    hmap = heatmap!(ax, xs,ys,zs, colormap = :plasma)
    for i in 1:n, j in 1:m
        txtcolor = mtx[i, j] < 0.2 ? :white : :black
        text!(ax, "$(round(mtx[i,j], digits = 2))", position = (j, i),
              color = txtcolor, align = (:center, :center))
    end
    Colorbar(fig[1, 2], hmap; label = "values", width = 15, ticksize = 15)
    ax.xticklabelrotation = π / 2
    ax.xticklabelalign = (:right, :center)

    hidedecorations!(ax; label = false,
                     ticklabels = false,
                     ticks = false,
                     grid = true,
                     minorgrid = true,
                     minorticks = true)
    save("$(filename)_$(now()).png", fig)
end

"""
比较整合前后cluster距离的变化

"""
function dist_compare(before,after;filename="dist_compare")
    CairoMakie.activate!()
    set_theme!(mytheme)

    fig = Figure()
    ax1 = Axis(fig[1, 1], xticks = ([1,2], ["before","after"]) )
    xlims!(ax1,0,3)
    score1 = before
    score2 = after
    # score2 .+= rand(3:0.2:4,30)
    n = size(after)[end]

    score = vcat(score1,score2)
    x = repeat([1,2],inner=n)
    for i in 1:n
        lines!(ax1, [1,2], [score1[i],score2[i]],color=(:grey,0.5))
    end
    scatter!(ax1, x, score, color = score)

    ax2 = Axis(fig[1,2], xticks = ([1,2], ["before","after"]) )

    boxplot!(ax2, fill(1,n), score1; orientation=:vertical,
             strokecolor = :black, strokewidth = 1, label = "before")
    boxplot!(ax2, fill(2,n), score2; orientation=:vertical,
             strokecolor = :black, strokewidth = 1, label = "after")

    ax3 = Axis(fig[1,3], xticks = ([1,2], ["before","after"]) )

    boxplot!(ax3, fill(1,n), score1; orientation=:vertical,
             whiskerwidth = 0.5,
             strokecolor = :black, strokewidth = 1, label = "before")
    boxplot!(ax3, fill(2,n), score2; orientation=:vertical,
             whiskerwidth = 0.5,
             strokecolor = :black, strokewidth = 1, label = "after")

    x = x .+ randn(2*n) ./ 10

    for i in 1:n
        lines!(ax3, [x[i],x[i+n]], [score1[i],score2[i]],color=(:grey,0.5))
    end
    scatter!(ax3, x, score, color = score)

    hideydecorations!(ax3)
    hideydecorations!(ax2)
    hideydecorations!(ax1)
    save("$(filename)_$(now()).png", fig, resolution=(1200,400))
end
function dist_compare(ds::Matrix,
                      ns = nothing;save_obj = true,
                      filename="dist_compare")
 CairoMakie.activate!()
 set_theme!(mytheme)

    fig = Figure()

    (yn,xn) = size(ds)
    if !(typeof(ns) <: Nothing)
        @assert length(ns) == size(ds,2)
    else
        ns = 1:xn
    end
    ax1 = Axis(fig[1, 1], xticks = ([1:xn...], string.(ns)) )
    ax2 = Axis(fig[1,2],xticks = ([1:xn...],string.(ns)))
    ax3 = Axis(fig[1,3], xticks = ([1:xn...],string.(ns)) )
    xlims!(ax1,0,xn + 1)
    xlims!(ax2,0,xn + 1)
    xlims!(ax3,0,xn + 1)

    score = ds |> vec

    x = repeat([1:xn...],inner=yn)

    for i in 1:yn
        lines!(ax1, [1:xn...],  ds[i,:],color=(:grey,0.5))
    end
    scatter!(ax1, x, score, color = score)

    # ax2
    for i in 1:xn
        boxplot!(ax2, fill(i,yn), ds[:,i]; orientation=:vertical,
                 strokecolor = :black, strokewidth = 1)
    end

    # ax3
    for i in 1:xn
        boxplot!(ax3, fill(i,yn),  ds[:,i]; orientation=:vertical,
                 whiskerwidth = 0.5,
                 strokecolor = :black, strokewidth = 1 )
    end
    x = x .+ randn(xn*yn) ./ 10
    for i in 1:yn
        lines!(ax3, [1:xn...],  ds[i,:],color=(:grey,0.5))
    end
    scatter!(ax3, x, score, color = score)

    hideydecorations!(ax3)
    hideydecorations!(ax2)
    hideydecorations!(ax1)
    if save_obj
        JLD2.@save "$(filename)_$(now()).fig" G
    end

    save("$(filename)_$(now()).png", fig, resolution=(1200,400))
end



"""
cluster_cor(data1,data2,info1,info2,label,comparePair::Pair;filename="Cor_in_cluster")
"""
function cluster_cor end
cluster_cor(data,info,label,comparePair::Pair;kws...) = cluster_cor(data,data,info,infolabel,comparePair::Pair;kws...)
cluster_cor(data1,data2,info1,info2,label,comparePair::Pair;filename="Cor_in_cluster") = begin
    @assert comparePair[1] ∈ info1 && comparePair[2] ∈ info2
    rowidx1 = indexin([comparePair[1]], info1 )
    rowidx2 = indexin([comparePair[2]], info2 )
    label = string.(label)
    tmp = unique(label)
    G = Figure( )

    if length(tmp) <= 4
        w = 2
    elseif length(tmp) <= 9
        w = 3
    else
        w = 4
    end
    res = similar(data1,length(tmp))
    for i in 1:length(tmp)
        colidx = findall( x -> x == tmp[i], label )
        x = data1[rowidx1,colidx]|>vec
        y = data2[rowidx2,colidx]|>vec

        keep = (x .> 0) .* ( y .> 0)
        if sum(keep) > 0

            x = x[keep]
            y = y[keep]
            corValue = cor(x,y) |> x -> round(x, digits=4)
            corValue = corValue == NaN ? 0 : corValue
            # @info corValue
            c = i % w == 0 ? w : (i % w)
            r = i % w == 0 ? i ÷ w : i ÷ w + 1
            p = Axis(G[ r, c ])
            p.title = "cor = $(corValue)"
            scatter!(p,x,y)
            res[i] = corValue
        end
    end
    save("$(filename)_$(now()).png",G)
    mean(res)
end


# todo
# 1. caculate cor of well known gene pairs (eg TF, makers, RNA vs protein)
# 2. caculate sd of well known marker gene for each cell type
