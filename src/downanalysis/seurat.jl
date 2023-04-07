"""
A julia wrap for R Seurat pipeline

given a expr::Matrix, and true label, \\
return the compare result of seurat clustering and true label.

```julia
seurat(expr,label;
    resolution = 0.5,
    nfeatures = 2000,
    filename="Seurat")
```
Result will save as PDF.

- normalized: whether the expr matrix have been normalized.
"""
seurat(expr,label;kws...) = seurat(expr,label = label;kws...)
function seurat(expr; neibor=20,
                label = nothing,
                resolution = 0.5, NPC = 30, selection_method = "disp",
                nfeatures = 2000, markersize = 2,
                filename="Seurat", normalized = true)
    # tranform args for R script
    filename = "$(filename)_$(now()).pdf"
    to_normal = normalized ? 0 : 1
    if nfeatures > size(expr,2)
        @info "nfeatures should not larger than sample size. \n It will be adjusted"
        @info nfeatures => size(expr,2)
        nfeatures =  size(expr,2)
    end
    if typeof(label) == Nothing
        addlabel = false
    else
        addlabel = true
    end
    # send the args to R
    @rput expr neibor filename addlabel label resolution selection_method nfeatures to_normal NPC markersize
         R"""
        library(Seurat)
        library(patchwork)
        library(ggplot2)
        df = list()
        rownames(expr) = paste0("gene",1:nrow(expr))
        colnames(expr) = paste0("cell",1:ncol(expr))
        df$expression = expr
        if (addlabel)  {
            df$celltype = label
        }
        df$cell_id = rownames(expr)
        pbmc <- CreateSeuratObject(counts = df$expression)

        if (to_normal == 1) {
            pbmc <- NormalizeData(pbmc,
            normalization.method = 'LogNormalize', scale.factor = 10000)
        }
        pbmc <- FindVariableFeatures(pbmc,
            selection.method = selection_method,
            nfeatures = nfeatures)
        pbmc <- ScaleData(pbmc )
        pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc), npcs=NPC)

        pbmc <- FindNeighbors(pbmc, reduction = "pca", dims = 1:NPC,  k.param = neibor)
        pbmc <- FindClusters(pbmc, resolution = resolution)

        pbmc <- RunUMAP(pbmc, dims = 1:NPC, min.dist = 0.1)
        p1 = DimPlot(pbmc, reduction = 'umap',group.by='seurat_clusters',pt.size=markersize)

        if ( addlabel ) {
            pbmc@meta.data$Ref_label = df$celltype
            p2 = DimPlot(pbmc, reduction = 'umap',group.by='Ref_label',pt.size=markersize)
            p = p1 /p2
        }

        ggsave(filename)

        label = pbmc@meta.data$seurat_clusters
        posi = pbmc[['umap']]@cell.embeddings
    """
    @rget  posi

    if  !addlabel
        @rget label
        return(label, posi'|>Array)
    end
        return(posi'|>Array)
end
#+end_src




function seuratNorm(expr; neibor=20,  NPC = 30, selection_method = "vst",
                nfeatures = 2000,   normalized = true)
    # tranform args for R script

    to_normal = normalized ? 0 : 1
    if nfeatures > size(expr,2)
        @info "nfeatures should not larger than sample size. \n It will be adjusted"
        @info nfeatures => size(expr,2)
        nfeatures =  size(expr,2)
    end

    # send the args to R
    @rput expr neibor selection_method nfeatures to_normal NPC
         R"""
        library(Seurat)
        library(patchwork)
        library(ggplot2)
        df = list()
        rownames(expr) = paste0("gene",1:nrow(expr))
        colnames(expr) = paste0("cell",1:ncol(expr))
        df$expression = expr
        if (addlabel)  {
            df$celltype = label
        }
        df$cell_id = rownames(expr)
        pbmc <- CreateSeuratObject(counts = df$expression)

        if (to_normal == 1) {
            pbmc <- NormalizeData(pbmc,
            normalization.method = 'LogNormalize', scale.factor = 10000)
        }
        pbmc <- FindVariableFeatures(pbmc,
            selection.method = selection_method,
            nfeatures = nfeatures)
  pbmc <- ScaleData(pbmc )
 pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
       # normalized_counts <- as.matrix(pbmc@assays$RNA@scale.data)
 normalized_counts <- as.matrix(t(pbmc@reductions$pca@cell.embeddings))
    """
    @rget normalized_counts
end
#+end_src
