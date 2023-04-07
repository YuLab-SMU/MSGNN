
function enrichGO(genelist;
                  ont = "MF",
                  type = :symbol)

@rput genelist  ont

R"""
library("clusterProfiler")
library("org.Hs.eg.db")

  bitr(genelist, "SYMBOL", "ENTREZID",'org.Hs.eg.db' ,drop = TRUE) -> b

ego <- enrichGO(gene          = b$ENTREZID,
                OrgDb         = org.Hs.eg.db,
                ont           = ont,
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.01,
                qvalueCutoff  = 0.05,
                readable      = TRUE)

goplot(ego)

"""
end

function compareGO(genelists::NamedTuple;
                  ont = "MF",
                  type = :symbol)

    R" l = list()";
    n = keys(genelists)
    for i in n
        R"""
    bitr($(genelists[i]), "SYMBOL", "ENTREZID",'org.Hs.eg.db' ,drop = TRUE) -> a
    l = append(l,  list( a$ENTREZID ))
            """
    end
    n = [n...]
    R"names(l) = $(n)"

@rput  ont
R"""
library("clusterProfiler")
library("org.Hs.eg.db")
ego <- compareCluster(geneCluster = l,
                  fun = enrichGO,
                OrgDb         = org.Hs.eg.db,
                ont           = ont,
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.01,
                qvalueCutoff  = 0.05,
                readable      = TRUE)


dotplot(ego)

"""
end
