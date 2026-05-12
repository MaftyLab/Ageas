library(SingleR)
library(Seurat)
library(SummarizedExperiment)

# 用于SingleR的自动注释和准确率计算
singleR_annotation_accuracy <- function(train_data, test_data, label_col = 'celltype') {
  # 制作reference和test数据的表达矩阵
  ref <- GetAssayData(train_data, slot = "data")
  test <- GetAssayData(test_data, slot = "data")

  # 获取reference的celltype标签
  ref_labels <- train_data@meta.data[[label_col]]

  # 运行SingleR
  singler_result <- SingleR(test = as.matrix(test),
                            ref = as.matrix(ref),
                            labels = ref_labels)

  # 获取预测结果
  predicted_labels <- singler_result$pruned.labels
  # 获取真实标签
  true_labels <- test_data@meta.data[[label_col]]

  # 计算accuracy（去除NA预测值）
  valid <- !is.na(predicted_labels)
  accuracy <- sum(predicted_labels[valid] == true_labels[valid]) / length(predicted_labels[valid])

  # 返回预测标签和准确率
  return(list(
    predicted_labels = predicted_labels,
    accuracy = accuracy
  ))
}

folder_name = c()
dataset_name = c()
acc = c()
dir1 = dir('F:\\AEGAS_analysis\\fivefold_rds')
pred_label_list = list()
for (i in dir1) {
  dir2 = dir(paste0('F:\\AEGAS_analysis\\fivefold_rds/',i))
  for (j in dir2) {
    folder = paste0(paste0('F:\\AEGAS_analysis\\fivefold_rds/',i,'/',j,'/'))
    train_data = readRDS(paste0(folder,'train.rds'))
    test_data = readRDS(paste0(folder,'test.rds'))
    train_data = NormalizeData(train_data)
    test_data = NormalizeData(test_data)
    res2 <- singleR_annotation_accuracy(train_data, test_data, label_col = 'celltype')
    folder_name = c(folder_name,j)
    dataset_name = c(dataset_name,i)
    acc = c(acc,res2$accuracy)
    pred_label_list[[j]] = res2$predicted_labels
  }
}
df_summary = data.frame(folder_name,dataset_name,acc)
df_summary$method = 'singlrR'
write.csv(df_summary,'F:\\AEGAS_analysis\\clf_summary/singler.csv',row.names = F,quote = F)
