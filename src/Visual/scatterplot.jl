
"""
给定坐标，绘制散点图
1. 如果提供 label，则按照 label 自动上色
1. 如果提供 colorvalue::Vector, 则把colorvalue 映射给node(比如marker gene表达)
1. 如果提供 compareOn， 则把对应node的连接起来
"""
function PlotScatter end
function plotScatter(Posi;
                     label=nothing, colorSet=prettycolor6,
                     cmap=:blues, filename = "filename",
                     colorvalue=nothing,markersize=10, backend = :Cairo,
                     size = (800,800), L2norm = false,
                     compareOn=nothing, return_fig = false, save_obj = true, kws...)
    if backend == :Cairo
        CairoMakie.activate!()
        ft = "png"
    elseif backend ==:pdf
        ft = "pdf"
    else
        GLMakie.activate!()
        ft = "jpg"
    end
    set_theme!(mytheme)
    if L2norm
        Posi = Posi ./ norm(Posi,1)
    end
    G = Figure()
    P_1 =  Axis(G[1,1])

    if isnothing(label) && isnothing(colorvalue)
        @views scatter!(P_1, Posi[1,:],  Posi[2,:], markersize = markersize)
    elseif !isnothing(label) && isnothing(colorvalue)
        color_scheme = getColorForLabel(label, colorSet)
        # scatter
        for i in keys(color_scheme)
            j = label .== i
            @views  scatter!(P_1,Posi[1,j],
                             Posi[2,j],
                             color=color_scheme[i],label="$(i)",
                             markersize = markersize )
        end
        G[1, 2] = Legend(G, P_1, "Cell Type", framevisible = false)
    else
        # scatter!(ax, x, y; color=diagValues, markersize=3 * abs.(diagValues),
        # colorrange=(adjmin, adjmax), colormap=cmap)
        @views scatter!(P_1, Posi[1,:], Posi[2,:],
                        color=colorvalue,
                        colormap=cmap, markersize = markersize  )
        # G[1, 2] = Makie.Colorbar(G, P_1, height = Relative(0.75), tickwidth = 2,
        # tickalign = 1, width = 14, ticksize = 14)
        #
        Colorbar(G[1, 2], label = "color", colormap = cmap)
    end

    if !isnothing(compareOn)
        @views Threads.@threads for (h,l) in compareOn
            for (x,y) in zip(h, l)
                lines!(P_1, Posi[1,[x,y]],Posi[2,[x,y]],color = :green, markersize = markersize )
                scatter!(P_1, [Posi[1,x]], [Posi[2,x]],color=:red, markersize = markersize)
                scatter!(P_1, [Posi[1,y]], [Posi[2,y]],color=:blue, markersize = markersize)
            end
        end
    end

    save("$(filename)_$(now()).$(ft)",G,resolution=size )
    if return_fig
        G
    end
end
plotScatter(Posi,label; kws...) = plotScatter(Posi;label=label, kws...)
