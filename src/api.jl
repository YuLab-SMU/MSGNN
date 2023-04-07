function cluster! end

function PCA! end

function umap! end

function leiden! end

function buildGraph! end

function plotCluster end
function graphLayout! end
function plotGraph end

"""
Marker cmap plot; show which part of cell is high expression

plotCluster(d,colorvalue=d.feature[1,:])
plotCluster(d,colorvalue=d.feature[1,:],
    compareOn = Pair_idx(d.feature,3=>4) )
plotCluster(d,colorvalue=d.feature[1,:],
    compareOn = Pair_idx(d.feature,"gene3"=>"gene4") )

plotCluster(d,colorvalue=d.feature[1,:])

linePair(d, 1=>3)
lineHvsL(d, ["gene1","gene12"])
"""
function lineHvsL end
function linePair end
