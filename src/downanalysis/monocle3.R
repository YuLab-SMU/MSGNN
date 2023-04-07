

library(ggplot2)
library(monocle3)
library(SingleCellExperiment)

# #start_ids <- unique(unlist(lapply(all_lineages_inter,function(e) e[1])))
# cell_info_=cell_info
# start_ids_=start_ids
# num_dim=20
# use_partition_=F

monocle3_run <- function(e_count,
                         cell_info_ = cell_info,
                         start_ids_ = start_ids,
                         num_dim = 20,
                         use_partition_ = F,
                         norm_method_ = "log",
                         gene_info = NULL) {
  # count横轴是细胞 纵轴是基因
  # 在monocle3里  log方法已经给矩阵加了1

  # gene label
  if (is.null(gene_info)) {
    gene_info <- data.frame(gene_short_name = colnames(e_count))
    rownames(gene_info) <- colnames(e_count)
  }

  # cell label
  if (length(cell_info_) == 0) {
    cell_info_ <- data.frame(cell_ids = rownames(e_count))
    rownames(cell_info_) <- rownames(e_count)
  }

  # 赋予对象
  cds <- monocle3::new_cell_data_set(t(e_count),
    cell_metadata = cell_info_,
    gene_metadata = gene_info
  )

  cds <- monocle3::preprocess_cds(cds, num_dim = num_dim, norm_method = norm_method_)
  # cds <- monocle3::preprocess_cds(cds, num_dim = 20, norm_method = "log")
  # plot(reducedDim(cds)[,1:2])
  # plot_pc_variance_explained(cds)
  # 原表达矩阵在10的前面就已经达到谷值了
  cds <- monocle3::reduce_dimension(cds, reduction_method = "UMAP")
  # plot(reducedDim(cds2,"UMAP"))

  cds <- monocle3::cluster_cells(cds, reduction_method = "UMAP")

  cds <- monocle3::learn_graph(cds, close_loop = F, use_partition = use_partition_)
  # 默认use_partition=T,发现全部细胞跑的时候有一部分没有伪时间，是灰色的，
  # 按照下面issue的某一位的猜测 是该分区内没有祖先节点，没有分配伪时间
  # 所以就不用分区来辅助轨迹推断 设置了use_partition=F
  # 但是该issue里又说  有开发版的monocle3已经解决这个问题了？？？？
  # https://github.com/cole-trapnell-lab/monocle3/issues/130


  if (length(start_ids_) == 0) {
    start_ids_ <- rownames(cell_info_)[1]
  }

  cds <- monocle3::order_cells(cds, root_cells = start_ids_)

  # pseudotime <- monocle3::pseudotime(cds)
}

# add label info
cell_info <- data.frame(label = label)
# add cell id into cell info
rownames(cell_info) <- 1:8752

# add cell id into RNA matrix
colnames(tmpd3) <- 1:8752
# add feature name into RNA matrix
rownames(tmpd3) <- 1:16

# data is cell X feature
cds <- monocle3_run(
  e_count = t(tmpd3), cell_info_ = cell_info,
  num_dim = 10, norm_method_ = "none",
  gene_info = NULL, start_ids_ = NULL
)

monocle3_plot(cds, label = "label")

## library(dplyr)
## ggplot_monocle_cds(cds,label="label",cell_info = cell_info)
##
monocle3_plot <- function(cds, label = "pseudotime") {
  # colData(cds)
  plot_cells(cds,
    color_cells_by = label,
    label_groups_by_cluster = FALSE,
    label_leaves = FALSE,
    label_branch_points = FALSE, ,
    group_label_size = 5,
    cell_size = 2.5
  )
}
