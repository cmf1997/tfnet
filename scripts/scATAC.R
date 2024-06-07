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


# ---------------------- GSE129785 map in hg19 ---------------------- #
# cluster1 Naive CD4 T
# cluster2 Th17
# cluster3 Tfh
# cluster4 Treg
# cluster5 Naive CD8 T
# cluster6 Th1
# cluster7 Memory CD8 T
# cluster8 CD8 TEx
# cluster9 Effector CD8 T

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

GSE129785 <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks",
  meta.data = metadata
)

Idents(GSE129785) <- GSE129785$Clusters
annotations = c("Naive CD4 T", "Th17", "Tfh", "Treg", "Naive CD8 T", "Th1", "Memory CD8 T", "CD8 TEx", "Effector CD8 T")
names(annotations) <- sort(levels(GSE129785))
GSE129785 <- RenameIdents(GSE129785, annotations)

DefaultAssay(GSE129785)

GSE129785[['UMAP']] <- CreateDimReducObject(embeddings = as.matrix(metadata[c('UMAP1','UMAP2')]), key = "UMAP_", global = T, assay = "peaks")


# ---------------------- save/load Seurat ---------------------- #
#save(GSE129785,file="/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/GSE129785.Rdata")
load("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/GSE129785.Rdata")


# ---------------- SplitFragments for bw signal ----------------#
DefaultAssay(GSE129785) <- 'peaks'
SplitFragments(
  object = GSE129785,
  assay = "peaks",
  idents = levels(GSE129785),
  outdir = "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/GSE129785/fragments/split/",
  append = TRUE,
  buffer_length = 256L,
  verbose = TRUE
)


# ---------------------- DA analysis ---------------------- #
da_peaks_all <- FindAllMarkers(
  object = GSE129785,
  #test.use = 'LR',
  test.use = 'wilcox',
  latent.vars = 'nCount_peaks'
)
save(da_peaks_all,file="/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/GSE129785.da_peaks_all.Rdata")
write.csv(da_peaks_all, "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/GSE129785.da_peaks.all.csv", row.names=TRUE)


# ---------------------- plot ---------------------- #
# dimplot
GSE129785_Dimplot <- DimPlot(GSE129785, reduction = "UMAP")
ggsave("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/R/GSE129785/GSE129785.dimplot.pdf", GSE129785_Dimplot , width = 8, height = 5)


GSE129785_Dimplot <- DimPlot(GSE129785, reduction = "UMAP", group.by = "orig.ident" )
ggsave("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/R/GSE129785/GSE129785.dimplot.patient.pdf", GSE129785_Dimplot, width = 8, height = 5)


# plot cell distribution heatmap
heatmap_data <- as.matrix(table(GSE129785$Group, Idents(GSE129785)))
GSE129785_heat <- pheatmap(heatmap_data, cluster_rows=F, cluster_cols=F, border_color=NA, display_numbers=T, number_format="%.0f", cellwidth=40, cellheight=20)
ggsave("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/R/GSE129785/GSE129785.heatmap.cell.dist.pdf", GSE129785_heat, width = 10, height = 10)

# only colnames contail Tcell
GSE129785_heat <- pheatmap(heatmap_data[grepl("Tcell", rownames(heatmap_data)),], cluster_rows=F, cluster_cols=F, border_color=NA, display_numbers=T, number_format="%.0f", cellwidth=40, cellheight=20)


# plot cell number and ratio
cell_counts_df = as.data.frame(table(Idents(GSE129785)))
colnames(cell_counts_df) <- c("Group", "Cell_Count")
count_plot <- ggplot(cell_counts_df, aes(x = Group, y = Cell_Count, fill=Group)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Cell Counts by Group",
       x = "Group",
       y = "Cell Count") +
  theme(plot.title = element_text(hjust = 0.5),
  legend.position="right",
  legend.title = element_blank()) + theme(axis.text.x = element_text(angle = 90)) + theme(legend.position="none")

ggsave("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/R/GSE129785/GSE129785.cell.dist.pdf", count_plot)















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
DefaultAssay(GSE181062) <- 'peaks'
SplitFragments(
  object = GSE181062,
  assay = "peaks",
  idents = levels(GSE181062),
  outdir = "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/GSE181062/fragments/split/",
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
save(da_peaks_2,file="/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/GSE181062.da_peaks.Rdata")
write.csv(da_peaks_2, "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/GSE181062.da_peaks.csv", row.names=TRUE)

head(da_peaks_1)

da_peaks_all <- FindAllMarkers(
  object = GSE181062,
  #test.use = 'LR',
  test.use = 'wilcox',
  latent.vars = 'nCount_peaks'
)
save(da_peaks_all,file="/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/GSE181062.da_peaks_all.Rdata")
write.csv(da_peaks_all, "/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/GSE181062.da_peaks.all.csv", row.names=TRUE)
#load(file="/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/Rdata/GSE181062.da_peaks_all.Rdata")






# ---------------------- plot ---------------------- #
# dimplot
GSE181062_Dimplot <- DimPlot(GSE181062, reduction = "UMAP")
ggsave("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/R/GSE181062/GSE181062.dimplot.pdf", GSE181062_Dimplot , width = 8, height = 5)

GSE181062_Dimplot <- DimPlot(GSE181062, reduction = "UMAP", group.by = "orig.ident" )
ggsave("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/R/GSE181062/GSE181062.dimplot.patient.pdf", GSE181062_Dimplot, width = 8, height = 5)

# plot cell distribution heatmap
GSE181062$sample_type <- paste(GSE181062$orig.ident, GSE181062$sampletype, sep="-")
heatmap_data <- table(GSE181062$sample_type, Idents(GSE181062))
GSE181062_heat <- pheatmap(heatmap_data, cluster_rows=F, cluster_cols=F, border_color=NA, display_numbers=T, number_format="%.0f", cellwidth=40, cellheight=20)
ggsave("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/R/GSE181062/GSE181062.heatmap.cell.dist.pdf", GSE181062_heat)


# valcano
cut_off_pvalue = 0.000001
cut_off_logFC = 1.6
da_peaks_2$change <- ifelse(da_peaks_2$p_val < cut_off_pvalue & abs(da_peaks_2$avg_log2FC) >= cut_off_logFC,
                     ifelse(da_peaks_2$avg_log2FC> cut_off_logFC ,'Up','Down'),
                     'Stable')
volcano_plot <- ggplot(
  da_peaks_2, aes(x = avg_log2FC, y = -log10(p_val), colour=change)) +
  geom_point(alpha=0.4, size=2, shape=16) +
  scale_color_manual(values=c("#546de5", "#d2dae2","#ff4757"))+
  geom_vline(xintercept=c(-cut_off_logFC, cut_off_logFC),lty=4,col="black",lwd=0.8) +
  geom_hline(yintercept = -log10(cut_off_pvalue),lty=4,col="black",lwd=0.8) +
  labs(x="log2(fold change)",
       y="-log10 (p-value)")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5),
        legend.position="right",
        legend.title = element_blank()) +
  ggtitle("Effector vs Naive CD8 T")

ggsave("volcano.eff.naive.cd8.pdf", volcano_plot)


# plot cell number and ratio
cell_counts_df = as.data.frame(table(Idents(GSE181062)))
colnames(cell_counts_df) <- c("Group", "Cell_Count")
count_plot <- ggplot(cell_counts_df, aes(x = Group, y = Cell_Count, fill=Group)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Cell Counts by Group",
       x = "Group",
       y = "Cell Count") +
  theme(plot.title = element_text(hjust = 0.5),
  legend.position="right",
  legend.title = element_blank()) + theme(axis.text.x = element_text(angle = 90)) + theme(legend.position="none")

ggsave("/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/R/GSE181062/GSE181062.cell.dist.pdf", count_plot)
