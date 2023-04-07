
# experiment record
# do not include or run this

# plot 2 umap picture in a 3D tube, and line the point.
x = randn(20)
y = randn(20)
z0,z1 = 0, 1
add_layer!(ax, x,y,z::Real;color=:gray) = begin
    cfx = (maximum(x) - minimum(x)) * 0.1
    cfy = (maximum(y) - minimum(y)) * 0.1
    max_x,max_y = maximum(x) + cfx, maximum(y) + cfy
    min_x,min_y = minimum(x) - cfx, minimum(y) - cfy
    p1 = Point3(min_x, min_y,z)
    p2 = Point3(min_x, max_y,z)
    p3 = Point3(max_x, max_y,z)
    p4 = Point3(max_x, min_y,z)
    # mesh!(ax1, [p1, p2, p3], color = (color,0.1), shading = false)
    # mesh!(ax1, [p1, p4, p3], color = (color,0.1), shading = false)
    lines!(ax1,[p1,p2,p3,p4,p1],color = (:black,0.2))
end

fig = Figure(;show_axis = false )
ax1 = Axis3(fig[1, 1]; aspect=(1, 1, 1), perspectiveness=0.5)
hidedecorations!.(ax1)
hidespines!(ax1)
add_layer!(ax1, x,y,z0 )
add_layer!(ax1, x,y,z1 )

scatter!(ax1, x, y, [z0 for _ in x],markersize=30)
scatter!(ax1, x, y, [z1 for _ in x],markersize=30)


lines!(ax1, [x[1],x[2]], [y[1],y[2]], [0,1])




import GLMakie
z0,z1 = 0, 1
add_layer!(ax, x,y,z::Real;color=:gray) = begin
    cfx = (maximum(x) - minimum(x)) * 0.1
    cfy = (maximum(y) - minimum(y)) * 0.1
    max_x,max_y = maximum(x) + cfx, maximum(y) + cfy
    min_x,min_y = minimum(x) - cfx, minimum(y) - cfy
    p1 = Point3(min_x, min_y,z)
    p2 = Point3(min_x, max_y,z)
    p3 = Point3(max_x, max_y,z)
    p4 = Point3(max_x, min_y,z)
    # mesh!(ax1, [p1, p2, p3], color = (color,0.1), shading = false)
    # mesh!(ax1, [p1, p4, p3], color = (color,0.1), shading = false)
    GLMakie.lines!(ax1,[p1,p2,p3,p4,p1],color = (:black,0.2))
end

fig = GLMakie.Figure(;show_axis = false )
ax1 = GLMakie.Axis3(fig[1, 1]; aspect=(1, 1, 1), perspectiveness=0.5)
# GLMakie.hidedecorations!.(ax1)
GLMakie.hidespines!(ax1)
add_layer!(ax1, x,y,z0 )
add_layer!(ax1, x1,y1,z1 )
colors = scGAE.getColorForLabel(label,prettycolor6)

GLMakie.scatter!(ax1, x, y, [z0 for _ in x],markersize=8,
         color=[colors[i] for i in label])
GLMakie.scatter!(ax1, x1, y1, [z1 for _ in x],markersize=8,
         color=[colors[i] for i in label])
GLMakie.save("test.png",fig)
