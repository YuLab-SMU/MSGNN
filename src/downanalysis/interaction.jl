
import GLMakie
import Makie
"""
# mouse selector

```
df = rand(12,2)

G = Figure()
p1 = Axis(G[1,1])
plot!(p1,df[:,1],df[:,2])

(G,select) =  cellSelect(G)
G;
# wait for the mouse interaction
idx = BitVector(repeat([false],size(df,1)))
for i in 1:size(df,1)
   idx[i] = checkinside(df[i,1],df[i,2],select)
end

plot!(G.content[2], df[idx,1],df[idx,2],color = :red)

```
"""
function cellSelect(G)
    GLMakie.activate!()
    p1 = G.content[1]
    p2 = Axis(G[1,2])
    # remove the interaction by name
    Makie.MakieLayout.deregister_interaction!(p1, :rectanglezoom)
    me = Makie.MakieLayout.addmouseevents!(p1.scene)
    ke = Makie.MakieLayout.addmouseevents!(p1.scene)
    # global var for value record
    # global debug_e
    global pos = Vector{Point}()
    # record the position of mouse when drag
    on(me.obs) do obs
        # global debug_e = obs
        if obs.type === Makie.MakieLayout.MouseEventTypes.leftdrag
            push!(pos,obs.data)
        end
    end

    # clear the position when press l
    on(p1.scene.events.keyboardbutton) do button
        if Makie.ispressed(p1.scene, Keyboard.l)
            @info "clear"
            global pos = Vector{Point}()
        end
    end

    # plot the position on p2 press p
    on(p2.scene.events.keyboardbutton) do button
        if Makie.ispressed(p2.scene, Keyboard.p)
            plot!(p2,Point.(pos))
        end
    end
    return (G, pos)
end



# check if a point x,y in the shape
function checkinside(x,y,pos)
    x, y = Float32.(x),Float32.(y)
    if Point(x,y) in pos
        return true
    end

    xs = [i[1] for i in pos]
    ys = [i[2] for i in pos]
    mx,my = maximum(xs), maximum(ys)

    # ignore the point lied in top or bottom line.
    if ys in [ maximum(ys),  minimum(ys)]
        return false
    end

    # check if have even cross
    counts = 0
    idx = xs .> x
    pos1 = pos[idx]
    for i in 1:length(pos1)-1
        if ((pos1[i][2] - y ) * (pos1[i+1][2] - y)) < 0
            counts += 1
        end
    end
    return !(counts % 2 == 0)
end



"""
圈出细胞，然后给出这类细胞的marker
"""
function select(G1::Figure)
    GLMakie.activate!()
    G = G1
    p1 = G.content[1]
    # p2 = Axis(G[1,2])
    @info "pick the data point from the plot"
    @info "c for clear the picture"
    @info "z for undo"
    @info "r for add to record"
    @info "d for delete last record"
    # remove the interaction by name
    :rectanglezoom in keys(p1.interactions) &&
        Makie.MakieLayout.deregister_interaction!(p1, :rectanglezoom)
    me = Makie.MakieLayout.addmouseevents!(p1.scene)
    ke = Makie.MakieLayout.addmouseevents!(p1.scene)
    # global var for value record
    # global debug_e
    pos1 = Vector{Point}()
    res = []
    sc = []
    # record the position of mouse when drag
    on(me.obs) do obs
        # global debug_e = obs
        if obs.type === Makie.MakieLayout.MouseEventTypes.leftdrag
            push!(pos1,obs.data)
            tmp = plot!(p1,Point.(pos1), markersize = 3)
            push!(sc,tmp)
        end
    end

    # clear the position when press r
    on(p1.scene.events.keyboardbutton) do button
        if Makie.ispressed(p1.scene, Keyboard.c)
            print("\rclear")
            pos1 = Vector{Point}()
            if length(sc) > 0
            for i in sc
                delete!(p1, i)
            end
            end
            sc = []
        end
    end

    # undo when press z
    on(p1.scene.events.keyboardbutton) do button
        if Makie.ispressed(p1.scene, Keyboard.z)
            pos1 = Vector{Point}()
            if sc != []
                print("\rundo")
                delete!(p1, sc[end])
                pop!(sc)
            end
        end
    end

    # clear the position when press r
    on(p1.scene.events.keyboardbutton) do button
        if Makie.ispressed(p1.scene, Keyboard.r)
            print("\rrecord")
            push!(res,pos1)
            pos1 = Vector{Point}()
        end
    end

    # on(p1.scene.events.keyboardbutton) do button
    #     if Makie.ispressed(p1.scene, Keyboard.s)
    #         @info "stop record"
    #         pos1 = Vector{Point}()
    #     end
    # end

    on(p1.scene.events.keyboardbutton) do button
        if Makie.ispressed(p1.scene, Keyboard.d)
            if length(res) > 0
            print("\rdelete last record")
            @info length(pop!(res))
            else
                print("\rrecord is empty")
            end
        end
    end
    # plot the position on p2 press p
    # on(p2.scene.events.keyboardbutton) do button
    #     if Makie.ispressed(p2.scene, Keyboard.p)
    #         plot!(p2,Point.(pos1))
    #     end
    # end

    return res
end

# G, ax, p1 = scatter(rand(32),rand(32))

# a =  select(G)


# emb = scGAE.umap(dt,dims=2)
# G = plotScatter(emb,label=label)
# a = select(G)
# idx1 = ((x,y) ->  checkinside(x,y,a[1])).(emb[1,:],emb[2,:])
# idx2 = ((x,y) ->  checkinside(x,y,a[2])).(emb[1,:],emb[2,:])
# res = FindMarkers(expr, idx1, idx2)
