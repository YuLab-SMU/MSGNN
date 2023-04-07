##########
# theme  #
##########

mytheme = Attributes(
    Axis = (
        spinewidth = 2,
        yticklabelsize = 25,
        xticklabelsize = 25
    )
)

"""
return the color scheme dict for the given label
"""
function getColorForLabel(label,colorSet)
    a_ =  unique(label)
    if length(a_) > length(colorSet)
        Random.seed!(42)
        n =  [RGBf(rand(3)...) for _ in 1:(length(a_) - length(colorSet))]
        colorSet = vcat(colorSet,n)
    end
    color_scheme = Dict()
    Random.seed!(42)
    for i in 1:length(a_)
        color_scheme[a_[i]] = colorSet[i]
    end
    color_scheme
end


"""
plot the line between selected points and connect
those points with high and low value

- hls: the pairs or the vectors Indicates
which is the high points and which is the low point
Vector{Vector} or Vector{Pair} or Pair.

eg: [[3,4,5],[12,45,22]], [3=>12,4=>45,5=>22]
"""
function lineHighAndLow(posi::Matrix,hls::Union{Pair, Vector};
                        filename="filename",markersize=5,
                        resolution = (800,800))
        CairoMakie.activate!()
     set_theme!(mytheme)
    for (h,l) in hls # the high and low cell idex
        cl =  repeat(["grey"],g.num_nodes)
        cl[h] .= "red"
        cl[l] .= "blue"
        G = Figure()
        p1 = G[1,1] |> Axis
        # plot the background points

        for i in ["grey"]
            k = cl .== i
            scatter!(p1,posi[k,1],posi[k,2],
                     color=i,
                     markersize=markersize)
        end

        # line the points
        for (x,y) in zip(h, l)
            lines!(p1, posi[[x,y],1],posi[[x,y],2],
                   color = :green )
        end

        # finally plot the points on the last layer.
        for i in ["red","blue"]
            k = cl .== i
            scatter!(p1,posi[k,1],posi[k,2],
                     color=i,
                     markersize=markersize)
        end

        save("$(filename)_$(now()).png",G, resolution = resolution)

    end
end


"""
show the line between the high and low point, based on the layout of a graph.

- compairOn
"""
linePair(g::GNNGraph,X,compairOn;topN::Int,filename="HvsL_compare",kws...) = begin
    pairs_idx = Pair_idx(X,compairOn,topN=topN)
    posi = graphToPosition(g)
    lineHighAndLow(posi,pairs_idx;kws...)
end
linePair(d::scData,compairOn;topN::Int,filename="HvsL_compare",kws...) = begin
    pairs_idx = Pair_idx(d.feature,compairOn,topN=topN)
    if isnothing(d.Layout2D)
        graphLayout!(d)
    end
    posi = g.Layout2D
    lineHighAndLow(posi,pairs_idx;kws...)
end


lineHvsL(g,X,compairOn;filename="HvsH_compare",topN::Int,kws...) = begin
    hls = HLidx(X,compairOn,topN=topN)
    posi = graphToPosition(g)
    lineHighAndLow(posi,hls;kws...)
end
lineHvsL(d::scData,compairOn;filename="HvsH_compare",topN::Int,kws...) = begin
    hls = HLidx(d.feature,compairOn,topN=topN)
    if isnothing(d.Layout2D)
        graphLayout!(d)
    end
    posi = g.Layout2D
    lineHighAndLow(posi,hls;kws...)
end


HLidx(X,compareOns::Vector{Int};topN=10) = begin
    s = size(X,2)
    res = []
    @assert s > 2 * topN
    for i in compareOns
        idx = sortperm(X[i,:])|>vec
        #  high => low
        push!(res, idx[end-topN+1:end] => idx[1:topN])
    end
    res
end
HLidx(X,compareOns::Vector{String};topN=10) = begin
    # check if gene in featureName
    FN = Int[] # store the feture idx
    for G in compareOns
        a = findfirst(x->x==G, d.featureName )
        if a != nothing
            push!(FN,a)
        else
            @info "drop $(G), because no conrespond feature"
        end
    end
    HLidx(X,FN;topN=topN)
end
Pair_idx(X,pairs::Vector{Pair{Int,Int}};topN=10) = begin
    s = size(X,2)
    res = []
    @assert s > 2 * topN
    for P in pairs
        idx1 = (sortperm(X[P[1],:])|>vec)[end-topN+1:end]
        idx2 = (sortperm(X[P[2],:])|>vec)[end-topN+1:end]
        a1 = setdiff(idx1, idx2)
        a2 = setdiff(idx2, idx1)
        if length(a1) > 0
            push!(res, a1 => a2)
        else
            @info "drop Pair $(P)"
        end
    end
    res
end
Pair_idx(X,pairs::Vector{Pair{String,String}};topN=10) = begin
  pair = Pair{Int,Int}[] #Vector{Pair{Int, Int}}[]
    for P in on
        a = findfirst(x->x==P[1], d.featureName )
        b = findfirst(x->x==P[2], d.featureName )
        if a != nothing && b != nothing
            push!(pair,a => b)
        else
            @info "drop $(P[1] => P[2]), because no conrespond genes"
        end
    end
    Pair_idx(X, pair;topN=topN)
end

"""
get border position of an ellipse
- cx: center point x
- cy: center point y
- rx: the radius on the x direction
- ry: the radius on the y direction
- θ: the angle of ellipse relate to x axis

"""
function getellipsepoints(cx, cy, rx=1, ry=1, θ = 0)
    t = range(0, 2*pi, length=100)
    ellipse_x_r = @. rx * cos(t)
    ellipse_y_r = @. ry * sin(t)
    R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
    r_ellipse = [ellipse_x_r ellipse_y_r] * R
    x = @. cx + r_ellipse[:,1]
    y = @. cy + r_ellipse[:,2]
    (x,y)
end

"""
fill the ellipse as the background of scatter
- xy: the position of scatter
- Adj: the adj matrix of nodes, from the KNN graph
- label: the node label.
- alpha_fill = 0.03: the alpha of ellipse
- alpha_point = 0.5: the alpha of the scatter.
- markersize=2, the markersize of scatter.
- colorSet = prettycolor6,
- filename
"""
function plotRange(xy, Adj, label;
                   alpha_fill = 0.03,
                   alpha_point = 0.5,
                   markersize=2,
                   colorSet = prettycolor6,
                   filename = "filename")
    G = Figure()
    P1 = Axis(G[1,1])
    # get the Degree ^ -1 matrix
    Adj += LinearAlgebra.I
    D = LinearAlgebra.diagm( sum(Adj,dims=2).^-1|>vec )
    # find the average direction and length of the neibor
    ave_direct =  xy * Adj * D
    angle = ave_direct[2,:] ./ (ave_direct[1,:] ./ (π/2) .+ eps())
    ave_dist =  abs.(ave_direct .- xy) #.* markersize
    #
    ave_dist .= 1
    #
    # color
    color_scheme = getColorForLabel(label, colorSet)
    # plot
    for i in unique(label)
        tmp = Vector{Point{2, Float32}}[]
        for j in [1:size(xy,2)...][label .== i]
            cx = xy[1,j]
            cy = xy[2,j]
            rx = ave_dist[1,j]#|>sqrt
            ry = ave_dist[2,j]#|>sqrt
            θ = angle[j]
            push!(tmp, Point2f.(getellipsepoints(cx, cy,rx, ry, θ)...))
        end

        poly!(P1, tmp;
              color=(color_scheme[i], alpha_fill))

         scatter!(P1,xy[1,label .== i],xy[2,label .== i],
                  color = (color_scheme[i],alpha_point),
                  markersize = markersize)
    end
    # save
    save("$(filename)_$(now()).png",G)
end

function plotRange(xy, label;kws...)
    g = KNNGraph(xy,5)
    adj = adjacency_matrix(g)
    plotRange(xy, adj, label;kws...)
end


"""
to fixed
"""
function plotCenter(xy, label;
                    markersize=10,
                    colorSet = prettycolor6,
                    filename = "filename")
    G = Figure()
    P1 = Axis(G[1,1])
    color_scheme = getColorForLabel(label, colorSet)
    # plot
    for i in unique(label)
        scatter!(P1,xy[1,label .== i],
                 xy[2,label .== i],
                 color = color_scheme[i],markersize = 1)
        centerXy = mean(xy[:,label .== i],dims=2)
        scatter!(P1,centerXy[1,:],centerXy[2,:],
                 markersize = markersize,
                 color = color_scheme[i])
    end
    # save
    save("$(filename)_$(now()).png",G)
end
