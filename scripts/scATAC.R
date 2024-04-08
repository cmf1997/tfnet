library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v75) # for hg19
library(EnsDb.Hsapiens.v86) # for hg38
library(ggplot2)
library(ggsci)
library(patchwork)
library(dplyr)

library(future)
plan("multicore", workers = 4)
options(future.globals.maxSize = 25 * 1024^3) # for 20 Gb RAM


# ---------------------- GSE181062 map in hg19 ---------------------- #
# ---------------------- start from mtx ---------------------- #
atac_matrix <- ReadMtx(
  mtx = "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/GSE129785/matrix.mtx.gz", 
  features = "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/GSE129785/features.tsv.gz",
  cells = "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/GSE129785/barcodes.tsv.gz",
  feature.column = 1,
  skip.cell = 1,
  skip.feature = 1
)
 
metadata <- read.csv(file="/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/GSE129785/meta_data.tsv.gz", header= T, sep="\t", row.names=1)

chrom_assay <- CreateChromatinAssay(
  counts = atac_matrix,
  sep = c("_", "_"),
  fragments = '/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/fragments/sort.rename.concat.fragments.tsv.gz',
)

GSE181062 <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks",
  meta.data = metadata
)

Idents(GSE181062) <- GSE181062$Clusters
annotations = c("Naive CD4 T", "Th17", "Tfh", "Treg", "Naive CD8 T", "Th1", "Memory CD8 T", "CD8 TEx", "Effector CD8 T")
names(annotations) <- sort(levels(GSE181062))
GSE181062 <- RenameIdents(GSE181062, annotations)

DefaultAssay(GSE181062)

GSE181062[['UMAP']] <- CreateDimReducObject(embeddings = as.matrix(metadata[c('UMAP1','UMAP2')]), key = "UMAP_", global = T, assay = "peaks")







# ---------------------- GSE181062 map in hg38 ---------------------- #
# ---------------------- start from fragments ---------------------- #
metadata <- read.csv(file = "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/GSE181062/GSE181062_RCC_eightPt_combined_scATAC.Tcells_MetaData.txt.gz", header = T, sep = "\t", row.names = 1)
peaks <- read.csv(file = "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/GSE181062/GSE181062_RCC_eightPt_combined_scATAC.Tcells_peakSet.txt.gz", header = T, sep = "\t")
fragments <- CreateFragmentObject("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/GSE181062/sort.rename.concat.fragments.tsv.gz")

atac_matrix <- FeatureMatrix(
    fragments = fragments,
    features = makeGRangesFromDataFrame(peaks),
    cell = rownames(metadata)
)

chrom_assay <- CreateChromatinAssay(
    counts = atac_matrix,
    sep = c("-", "-"),
    fragments = fragments,
)

GSE181062 <- CreateSeuratObject(
    counts = chrom_assay,
    assay = "peaks",
    meta.data = metadata
)

Idents(GSE181062) <- GSE181062$Clusters

annotations <- c("Dysfunctional CD8 T", "Effector CD8 T (blood/tumor)", "Effector CD8 T (tumor/blood)", "Naive CD4 T", "Effector CD8 T (tumor/normal adjacent)", "Dysfunctional CD8 T", "Dysfunctional CD8 T", "Memory/Effector CD4 T", "Naive CD4 T", "Early dysfunctional CD8 T", "Effector CD8 T (tumor)", "Tregs")
# annotations = c("Dysfunctional CD8 T", "Effector CD8 T", "Effector CD8 T", "Naive CD4 T", "Effector CD8 T", "Dysfunctional CD8 T", "Dysfunctional CD8 T", "Memory/Effector CD4 T", "Naive CD4 T", "Early dysfunctional CD8 T", "Effector CD8 T", "Tregs")

names(annotations) <- str_sort(levels(GSE181062), numeric = TRUE)
GSE181062 <- RenameIdents(GSE181062, annotations)
DefaultAssay(GSE181062)
GSE181062[["UMAP"]] <- CreateDimReducObject(embeddings = as.matrix(metadata[c("UMAP_Dimension_1", "UMAP_Dimension_2")]), key = "UMAP_", global = T, assay = "peaks")





# ---------------- SplitFragments for bw signal ----------------#
SplitFragments(
  object = GSE181062,
  assay = "peaks",
  group.by = "Clusters",
  idents = ccluster_vector <- paste0("Cluster", 1:9),
  outdir = "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/fragments/split/",
  #file.suffix = "",
  append = TRUE,
  buffer_length = 256L,
  verbose = TRUE
)


# ---------------------- DA analysis ---------------------- #
da_peaks_2 <- FindMarkers(
  object = GSE181062,
  ident.1 = "Effector CD8 T",  
  ident.2 = "Naive CD8 T",
  test.use = 'LR',
  latent.vars = 'nCount_peaks'
)
save(da_peaks_2,file="/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/da_peaks.2.Rdata")
write.csv(da_peaks_2, "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/da_peaks.2.csv", row.names=TRUE)

head(da_peaks_1)

da_peaks_all <- FindAllMarkers(
  object = GSE181062,
  test.use = 'LR',
  latent.vars = 'nCount_peaks'

)
#save(da_peaks_all,file="/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/da_peaks_all.Rdata")
load(file="/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/da_peaks_all.Rdata")