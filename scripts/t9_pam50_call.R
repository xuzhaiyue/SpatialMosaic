# T9-1 第二步：genefu PAM50 调用。各队列内部按基因中位数居中（pam50 模型 std="none"，居中由我们事先固定），设定不随结果改。
suppressMessages(library(genefu))
data(pam50); data(pam50.robust)  # molecular.subtyping(sbt.model="pam50") 内部使用 pam50.robust
o <- "results/t9/pam50"
for (cohort in c("TCGA-BRCA", "CPTAC-BRCA")) {
  x <- read.csv(file.path(o, paste0(cohort, "_PAM50_log2_expression.csv")), row.names = 1, check.names = FALSE)
  x <- as.matrix(x)
  stopifnot(setequal(colnames(x), rownames(pam50$centroids)))
  x <- sweep(x, 2, apply(x, 2, median), "-")
  annot <- data.frame(probe = colnames(x), EntrezGene.ID = pam50$centroids.map[colnames(x), "EntrezGene.ID"], row.names = colnames(x))
  res <- molecular.subtyping(sbt.model = "pam50", data = x, annot = annot, do.mapping = FALSE)
  out <- data.frame(patient_id = rownames(x), PAM50_genefu = as.character(res$subtype), res$subtype.proba, check.names = FALSE)
  write.csv(out, file.path(o, paste0(cohort, "_PAM50_genefu_calls.csv")), row.names = FALSE)
  cat(cohort, nrow(out), "\n"); print(table(out$PAM50_genefu))
}
