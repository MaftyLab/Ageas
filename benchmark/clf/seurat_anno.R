library(Seurat)
seurat_transfer_annotate <- function(train_data, test_data, label_col = 'celltype') {

  train_data = NormalizeData(train_data)
  train_data = ScaleData(train_data)
  train_data = FindVariableFeatures(train_data)
  train_data = RunPCA(train_data)


  train_data[[label_col]] <- as.factor(train_data@meta.data[[label_col]])

  anchors <- FindTransferAnchors(reference = train_data, query = test_data,
                                 dims = 1:30)

  predictions <- TransferData(anchorset = anchors,
                              refdata = train_data@meta.data[[label_col]],
                              dims = 1:30)

  test_data$predicted_celltype <- predictions$predicted.id

  true_labels <- test_data@meta.data[[label_col]]
  predicted_labels <- test_data$predicted_celltype
  accuracy <- sum(predicted_labels == true_labels,
                  na.rm = TRUE) / sum(!is.na(predicted_labels) & !is.na(true_labels))

  return(list(
    predicted_labels = predicted_labels,
    accuracy = accuracy
  ))
}

folder_name = c()
dataset_name = c()
acc = c()
dir1 = dir('/data/jiangjunyao/AEGAS_data/celltype annotation/AEGAS_anno/fivefold_rds')
pred_label_list = list()
for (i in dir1) {
  dir2 = dir(paste0('/data/jiangjunyao/AEGAS_data/celltype annotation/AEGAS_anno/fivefold_rds/',i))
  for (j in dir2) {
    folder = paste0(paste0('/data/jiangjunyao/AEGAS_data/celltype annotation/AEGAS_anno/fivefold_rds/',
                           i,'/',j,'/'))
    train_data = readRDS(paste0(folder,'train.rds'))
    test_data = readRDS(paste0(folder,'test.rds'))
    train_data = NormalizeData(train_data)
    test_data = NormalizeData(test_data)
    res2 <- seurat_transfer_annotate(train_data, test_data,
                                     label_col = 'celltype')
    folder_name = c(folder_name,j)
    dataset_name = c(dataset_name,i)
    acc = c(acc,res2$accuracy)
    pred_label_list[[j]] = res2$predicted_labels
  }
}
df_summary = data.frame(folder_name,dataset_name,acc)
df_summary$method = 'Seurat'
zheng68 = pred_label_list[1:5]
all_label = c()
for (i in zheng68) {
  all_label = c(all_label,i)
}
df_label = data.frame(names(all_label),all_label)
write.csv(df_summary,'/data/jiangjunyao/AEGAS_data/celltype annotation/anno_result/benchmark_method/seurat.csv',
          quote = F,row.names = F)
write.csv(df_label,'/data/jiangjunyao/fa_result/zheng68/seurat_pred.csv')
