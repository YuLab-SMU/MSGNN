
"""
Plot;  with or without label
- Scatter: raw data, filter data, PCA, UMAP | with or without label
- Graph: Adj
"""
function graphLayout!(d::scData;seed=42,n=3000,label = nothing,kws...)
    if isnothing(d.adj)
        @info "running umap! with default args. or you can run it manually"
        umap!(d;kws...)
    end
    Random.seed!(seed)

    @info "down-sample if node num > 3000"
    s = size(d.adj,1)
    if s > 3000
        X,label = downSample(d.adj,n=3000,label = label,kws...)
    end
    a = Spring(kws...)(Array(d.adj))
    d.layout2d = (zeros(Float32, 2, length(a)),label)

    @views Threads.@threads for i in 1:length(a)
        d.layout2d[1][1,i] = a[i][1]
        d.layout2d[1][2,i] = a[i][2]
    end  # size(posi) == (n,2)
    nothing
end

function plotGraph(d::scData;
                   colorSet = prettycolor6,
                   edge = true,
                   label = nothing,
                   markersize=5,
                   colorvalue = nothing,
                   filename = "filename",
                   resolution=(800, 600),kws...)

    if isnothing(d.layout2d)
        @info "running umap! with default args. or you can run it manually"
        graphLayout!(d;kws...)
    end

    (pos,label) = d.layout2d
    x = [i[1] for i in pos]
    y = [i[2] for i in pos]
    adjmin = minimum(adjacencyM)
    adjmax = maximum(adjacencyM)
    diagValues = diag(adjacencyM)
    segm, weights = getGraphEdges(adjacencyM, x, y)

    CairoMakie.activate!()
    set_theme!(mytheme)

    fig = Figure()
    ax = fig[1,1] |> Axis

    if edge
        linesegments!(ax,segm;
                      color=:grey,
                      linewidth=0.1,)
    end

    if label == nothing && colorvalue == nothing
        scatter!(ax,x, y,markersize=markersize)
    elseif colorvalue !== nothing
        scatter!(ax,x, y,markersize=markersize)
        @views scatter!(ax, x, y,
                        color=colorvalue,
                        colormap=:blues, markersize = markersize  )
        Colorbar(fig[1, 2], label = "color", colormap = :blues)
    else
        label = string.(label)
        color_scheme = getColorForLabel(label, colorSet)
        # scatter
        @views for i in keys(color_scheme)
            j = label .== i
            scatter!(ax, x[j], y[j],
                     markersize=markersize,
                     color=color_scheme[i],
                     label="$(i)" )
        end
        fig[1, 2] = Legend(fig, ax, "Cell Type", framevisible = false)
    end
    # scatter!(ax, x, y; color=diagValues, markersize=3 * abs.(diagValues),
        # colorrange=(adjmin, adjmax), colormap=cmap)
    save("$(filename)_$(now()).png",fig,resolution=resolution)
end



# """
# 给定一个GNNgraph 对象，计算 layout, 并绘散点图。
# """
# plotLayout(g,label;layout=Spring(),kws...) =
#     plotLayout(g;layout=layout,label = label,kws...)
# plotLayout(g;layout=Spring(),kws...) = begin
#     CairoMakie.activate!()
#     adjacencyM = adjacency_matrix(g)
#     a = layout(Array(  adjacencyM  ))
#     posi = zeros(Float32,2,length(a))
#     @views Threads.@threads for i in 1:length(a)
#         posi[1,i] = a[i][1]
#         posi[2,i] = a[i][2]
#     end
#     plotScatter(posi;kws...)
# end



"""
给定一个邻接矩阵, 及二维坐标， 计算需要连接的节点坐标及边的连接位置
"""
function getGraphEdges(adjMatrix, x, y)
    xyos = []
    weights = []
    for i in eachindex(x), j in i+1:length(x)
        if adjMatrix[i, j] != 0.0
            push!(xyos, [x[i], y[i]])
            push!(xyos, [x[j], y[j]])
            push!(weights, adjMatrix[i, j])
            push!(weights, adjMatrix[i, j])
        end
    end
    return (Point2f.(xyos), Float32.(weights))
end

"""
给定一个Graph，计算graph布局并绘制成图。
"""
plotGraph(g,label;kws...) = plotGraph(g;label = label,kws...)
function plotGraph(g;layout = Spring(),
                    colorSet = prettycolor6,edge = false,
                    label = nothing,markersize=5,
                    resolution=(800, 600),colorvalue = nothing,
                    filename = "filename" )
    CairoMakie.activate!()

    adjacencyM = adjacency_matrix(g)
    pos = layout(Array( adjacencyM ))
    x = [i[1] for i in pos]
    y = [i[2] for i in pos]
    adjmin = minimum(adjacencyM)
    adjmax = maximum(adjacencyM)
    diagValues = diag(adjacencyM)
    segm, weights = getGraphEdges(adjacencyM, x, y)

    # fig, ax, pltobj = linesegments(segm; color=weights, colormap=cmap,
    #     linewidth=abs.(weights) / 2, colorrange=(adjmin, adjmax),
    #     figure=(; resolution=(800, 600)),
    #     axis=(; aspect=DataAspect()))
    fig = Figure()
    set_theme!(mytheme)
    ax = fig[1,1] |> Axis
    if edge
        linesegments!(ax,segm;
                      color=:grey,
                      linewidth=0.1,)
        # figure=(; resolution=resolution),
        # axis=(; aspect=DataAspect()))
    end

    if label == nothing && colorvalue == nothing
        scatter!(ax,x, y,markersize=markersize)
    elseif colorvalue !== nothing
        scatter!(ax,x, y,markersize=markersize)
        @views scatter!(ax, x, y,
                        color=colorvalue,
                        colormap=:blues, markersize = markersize  )
        Colorbar(fig[1, 2], label = "color", colormap = :blues)
    else
        label = string.(label)
        color_scheme = getColorForLabel(label, colorSet)
        # scatter
        @views for i in keys(color_scheme)
            j = label .== i
            scatter!(ax, x[j], y[j],
                     markersize=markersize,
                     color=color_scheme[i],
                     label="$(i)" )
        end
        fig[1, 2] = Legend(fig, ax, "Cell Type", framevisible = false)
    end
    # scatter!(ax, x, y; color=diagValues, markersize=3 * abs.(diagValues),
        # colorrange=(adjmin, adjmax), colormap=cmap)
    save("$(filename)_$(now()).png",fig,resolution=resolution)
end
