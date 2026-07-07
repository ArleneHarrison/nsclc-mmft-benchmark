options(stringsAsFactors = FALSE)

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
if (length(file_arg) == 0) {
  stop("Run this script with Rscript so the --file argument is available.")
}
script_path <- normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE)
project_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
data_dir <- file.path(project_dir, "data")
result_dir <- file.path(project_dir, "result")
dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)

missing_tokens <- c("", "NA", "N/A", "Not Collected", "Not collected",
                    "Not Recorded In Database", "Not Assessed", "nan")

clean_missing <- function(x) {
  x <- trimws(as.character(x))
  x[x %in% missing_tokens] <- NA
  x
}

find_col <- function(df, pattern) {
  hit <- grep(pattern, names(df), value = TRUE, ignore.case = TRUE)
  if (length(hit) == 0) {
    stop(paste("Column not found for pattern:", pattern))
  }
  hit[1]
}

make_location <- function(df) {
  loc_cols <- grep("^Tumor Location", names(df), value = TRUE)
  loc <- rep(NA_character_, nrow(df))
  for (cc in loc_cols) {
    checked <- clean_missing(df[[cc]]) == "Checked"
    label <- sub("^Tumor Location \\(choice=", "", cc)
    label <- sub("\\)$", "", label)
    loc[!is.na(checked) & checked] <- label
  }
  loc
}

auc_ci <- function(response, score) {
  roc_obj <- pROC::roc(response, score, quiet = TRUE, direction = "<")
  ci <- as.numeric(pROC::ci.auc(roc_obj))
  c(auc = as.numeric(pROC::auc(roc_obj)), ci_low = ci[1], ci_high = ci[3])
}

calibration_table <- function(response, score, groups = 5) {
  qs <- unique(quantile(score, probs = seq(0, 1, length.out = groups + 1), na.rm = TRUE))
  grp <- cut(score, breaks = qs, include.lowest = TRUE, labels = FALSE)
  aggregate(data.frame(pred = score, obs = response), list(group = grp), function(z) mean(z, na.rm = TRUE))
}

decision_curve <- function(response, score, thresholds = seq(0.05, 0.8, by = 0.01)) {
  n <- length(response)
  prevalence <- mean(response)
  out <- lapply(thresholds, function(pt) {
    pred <- score >= pt
    tp <- sum(pred & response == 1)
    fp <- sum(pred & response == 0)
    net_model <- tp / n - fp / n * pt / (1 - pt)
    net_all <- prevalence - (1 - prevalence) * pt / (1 - pt)
    data.frame(threshold = pt, model = net_model, treat_all = net_all, treat_none = 0)
  })
  do.call(rbind, out)
}

vif_table <- function(model_data, vars) {
  x <- model.matrix(as.formula(paste("~", paste(vars, collapse = " + "))), data = model_data)[, -1, drop = FALSE]
  if (ncol(x) < 2) {
    return(data.frame(variable = colnames(x), VIF = NA_real_))
  }
  out <- lapply(seq_len(ncol(x)), function(i) {
    fit <- lm(x[, i] ~ x[, -i, drop = FALSE])
    r2 <- summary(fit)$r.squared
    data.frame(variable = colnames(x)[i], VIF = 1 / (1 - r2))
  })
  do.call(rbind, out)
}

hosmer_lemeshow <- function(response, score, groups = 5) {
  qs <- unique(quantile(score, probs = seq(0, 1, length.out = groups + 1), na.rm = TRUE))
  grp <- cut(score, breaks = qs, include.lowest = TRUE, labels = FALSE)
  tab <- aggregate(data.frame(obs = response, exp = score), list(group = grp), function(z) c(sum = sum(z), n = length(z)))
  obs_events <- tab$obs[, "sum"]
  n_group <- tab$obs[, "n"]
  exp_events <- tab$exp[, "sum"]
  exp_nonevents <- n_group - exp_events
  obs_nonevents <- n_group - obs_events
  stat <- sum((obs_events - exp_events)^2 / pmax(exp_events, 1e-8) +
                (obs_nonevents - exp_nonevents)^2 / pmax(exp_nonevents, 1e-8))
  df <- max(length(n_group) - 2, 1)
  data.frame(groups = length(n_group), statistic = stat, df = df, p = 1 - pchisq(stat, df))
}

threshold_metrics <- function(response, score, threshold) {
  pred <- ifelse(score >= threshold, 1, 0)
  tp <- sum(pred == 1 & response == 1)
  fp <- sum(pred == 1 & response == 0)
  tn <- sum(pred == 0 & response == 0)
  fn <- sum(pred == 0 & response == 1)
  data.frame(
    threshold = threshold,
    TP = tp, FP = fp, TN = tn, FN = fn,
    sensitivity = tp / (tp + fn),
    specificity = tn / (tn + fp),
    PPV = tp / max(tp + fp, 1),
    NPV = tn / max(tn + fn, 1),
    accuracy = (tp + tn) / length(response)
  )
}

bootstrap_validate_auc <- function(model_data, formula, apparent_auc, b = 200, seed = 20260701) {
  set.seed(seed)
  n <- nrow(model_data)
  optimism <- rep(NA_real_, b)
  for (i in seq_len(b)) {
    idx <- sample(seq_len(n), size = n, replace = TRUE)
    boot_dat <- model_data[idx, , drop = FALSE]
    if (length(unique(boot_dat$pn_positive)) < 2) next
    boot_fit <- tryCatch(glm(formula, data = boot_dat, family = binomial()), error = function(e) NULL)
    if (is.null(boot_fit)) next
    boot_pred <- tryCatch(as.numeric(predict(boot_fit, newdata = boot_dat, type = "response")), error = function(e) NULL)
    test_pred <- tryCatch(as.numeric(predict(boot_fit, newdata = model_data, type = "response")), error = function(e) NULL)
    if (is.null(boot_pred) || is.null(test_pred)) next
    boot_auc <- tryCatch(as.numeric(pROC::auc(pROC::roc(boot_dat$pn_positive, boot_pred, quiet = TRUE, direction = "<"))), error = function(e) NA_real_)
    test_auc <- tryCatch(as.numeric(pROC::auc(pROC::roc(model_data$pn_positive, test_pred, quiet = TRUE, direction = "<"))), error = function(e) NA_real_)
    optimism[i] <- boot_auc - test_auc
  }
  optimism <- optimism[is.finite(optimism)]
  data.frame(
    bootstrap_reps_requested = b,
    bootstrap_reps_used = length(optimism),
    apparent_auc = apparent_auc,
    mean_optimism = mean(optimism),
    optimism_corrected_auc = apparent_auc - mean(optimism)
  )
}

write_stage_report <- function(path, summary_lines, selected_base, perf, boot, threshold_out, hl, vifs, coef_table, cv_auc, cv_auc_se) {
  lines <- c(
    "# 公共数据库建模阶段性报告",
    "",
    "## 数据来源",
    "",
    "公共数据库：TCIA NSCLC Radiogenomics。",
    "",
    "公共库建模使用临床 CSV 与 AIM XML 影像语义标注。结局定义为病理淋巴结转移，pN0 = 0，pN1/pN2 = 1。",
    "",
    "## 分析队列",
    "",
    paste0("- ", summary_lines),
    "",
    "## 候选变量",
    "",
    "候选变量包括年龄、性别、吸烟史、病理类型、肿瘤位置，以及 AIM 标注解析得到的 CT 语义特征，例如中央/外周型、实性/部分实性、边缘毛刺或不规则、分叶、胸膜牵拉、血管集束、支气管充气征和肺气肿。",
    "",
    "## 建模方法",
    "",
    "采用 LASSO logistic regression 进行变量筛选，并用筛选变量拟合最终 logistic regression 模型。模型评价包括 AUC、Brier score、校准、Hosmer-Lemeshow test、DCA、VIF 和 bootstrap 内部验证。",
    "",
    "## 最终模型",
    "",
    paste0("选入变量：", paste(selected_base, collapse = ", ")),
    "",
    "### 模型系数",
    "",
    paste(capture.output(print(coef_table, row.names = FALSE)), collapse = "\n"),
    "",
    "## 模型表现",
    "",
    paste0("- Apparent AUC：", sprintf("%.3f", perf$auc), " (95% CI ", sprintf("%.3f", perf$ci_low), "-", sprintf("%.3f", perf$ci_high), ")"),
    paste0("- 5-fold LASSO CV AUC：", sprintf("%.3f", cv_auc), " ± ", sprintf("%.3f", cv_auc_se)),
    paste0("- Bootstrap optimism-corrected AUC：", sprintf("%.3f", boot$optimism_corrected_auc), "；平均 optimism：", sprintf("%.3f", boot$mean_optimism), "；有效重复：", boot$bootstrap_reps_used),
    paste0("- Brier score：", sprintf("%.3f", perf$brier)),
    paste0("- Hosmer-Lemeshow p：", sprintf("%.3f", hl$p)),
    "",
    "### 阈值分类表现",
    "",
    paste(capture.output(print(threshold_out, row.names = FALSE)), collapse = "\n"),
    "",
    "### 共线性",
    "",
    paste(capture.output(print(vifs, row.names = FALSE)), collapse = "\n"),
    "",
    "## 当前解释",
    "",
    "公共库模型存在中等预测信号，可作为前期模型开发基础。由于样本量和事件数有限，公共库模型不宜被描述为最终临床模型；后续应在本院队列中进行外部验证，并加入术前炎症指标进行增量价值分析。",
    "",
    "## 后续补充",
    "",
    "后期本院队列需要补充术前血常规、炎症指标、CT 语义特征、病理 pN 分期和淋巴结清扫质量。建议收集完成后运行 `scripts/analyze_hospital_validation.R`。"
  )
  writeLines(lines, path)
}

clinical <- read.csv(file.path(data_dir, "nsclc_radiogenomics_clinical.csv"), check.names = FALSE)
aim <- read.csv(file.path(data_dir, "nsclc_radiogenomics_aim_features.csv"), check.names = FALSE)

pn_col <- find_col(clinical, "^Pathological N stage$")
age_col <- find_col(clinical, "^Age at Histological Diagnosis$")
gender_col <- find_col(clinical, "^Gender$")
smoking_col <- find_col(clinical, "^Smoking status$")
histology_col <- find_col(clinical, "^Histology")

clinical$pn_raw <- clean_missing(clinical[[pn_col]])
clinical <- clinical[clinical$pn_raw %in% c("N0", "N1", "N2"), ]
clinical$pn_positive <- ifelse(clinical$pn_raw == "N0", 0, 1)
clinical$age <- as.numeric(clean_missing(clinical[[age_col]]))
clinical$male <- ifelse(clean_missing(clinical[[gender_col]]) == "Male", 1, 0)
clinical$ever_smoker <- ifelse(clean_missing(clinical[[smoking_col]]) %in% c("Former", "Current"), 1, 0)
clinical$squamous <- ifelse(grepl("Squamous", clean_missing(clinical[[histology_col]]), ignore.case = TRUE), 1, 0)
clinical$tumor_location <- make_location(clinical)
clinical$upper_lobe <- ifelse(clinical$tumor_location %in% c("RUL", "LUL"), 1, 0)

dat <- merge(clinical, aim, by = "Case ID", all = FALSE)
dat$central <- ifelse(dat$axial_location == "central", 1, 0)
dat$solid_or_partsolid <- dat$is_solid_or_partsolid

candidate_vars <- c(
  "age", "male", "ever_smoker", "squamous", "upper_lobe",
  "central", "solid_or_partsolid", "margin_spiculated_or_irregular",
  "margin_lobulated", "pleural_retraction", "attachment_to_pleura",
  "vascular_convergence", "air_bronchogram", "emphysema_present"
)
dat_model <- dat[, c("Case ID", "pn_positive", candidate_vars)]
dat_model <- na.omit(dat_model)

missingness <- do.call(rbind, lapply(c("pn_positive", candidate_vars), function(v) {
  data.frame(variable = v, missing = sum(is.na(dat[, v])), total = nrow(dat), missing_rate = sum(is.na(dat[, v])) / nrow(dat))
}))
write.csv(missingness, file.path(result_dir, "public_model_missingness.csv"), row.names = FALSE)

summary_lines <- c(
  paste("Radiogenomics clinical rows:", nrow(clinical)),
  paste("AIM feature rows:", nrow(aim)),
  paste("Merged analyzable rows:", nrow(dat_model)),
  paste("pN positive events:", sum(dat_model$pn_positive), "of", nrow(dat_model))
)

writeLines(summary_lines)
writeLines(summary_lines, file.path(result_dir, "public_model_summary.txt"))

univ <- do.call(rbind, lapply(candidate_vars, function(v) {
  f <- as.formula(paste("pn_positive ~", v))
  fit <- glm(f, data = dat_model, family = binomial())
  co <- summary(fit)$coefficients
  or <- exp(co[2, 1])
  ci <- exp(confint.default(fit)[2, ])
  data.frame(variable = v, OR = or, CI_low = ci[1], CI_high = ci[2], p = co[2, 4], row.names = NULL)
}))
write.csv(univ, file.path(result_dir, "univariable_public_model.csv"), row.names = FALSE)

x <- model.matrix(pn_positive ~ . - `Case ID`, data = dat_model)[, -1]
y <- dat_model$pn_positive
set.seed(20260701)
cvfit <- glmnet::cv.glmnet(x, y, family = "binomial", alpha = 1, type.measure = "auc", nfolds = 5)
cv_idx <- cvfit$index["min", 1]
cv_auc <- cvfit$cvm[cv_idx]
cv_auc_se <- cvfit$cvsd[cv_idx]
coef_min <- as.matrix(coef(cvfit, s = "lambda.min"))
selected <- rownames(coef_min)[coef_min[, 1] != 0]
selected <- setdiff(selected, "(Intercept)")
if (length(selected) == 0) {
  selected <- candidate_vars[order(univ$p)][1:min(3, length(candidate_vars))]
}

selected_base <- unique(sub("`", "", selected))
selected_base <- selected_base[selected_base %in% colnames(dat_model)]
if (length(selected_base) == 0) {
  selected_base <- candidate_vars[order(univ$p)][1:min(3, length(candidate_vars))]
}

final_formula <- as.formula(paste("pn_positive ~", paste(selected_base, collapse = " + ")))
final_fit <- glm(final_formula, data = dat_model, family = binomial())
dat_model$pred <- as.numeric(predict(final_fit, type = "response"))

coef_raw <- summary(final_fit)$coefficients
coef_ci <- confint.default(final_fit)
coef_table <- data.frame(
  variable = rownames(coef_raw),
  beta = coef_raw[, 1],
  OR = exp(coef_raw[, 1]),
  CI_low = exp(coef_ci[, 1]),
  CI_high = exp(coef_ci[, 2]),
  p = coef_raw[, 4],
  row.names = NULL
)
write.csv(coef_table, file.path(result_dir, "public_model_coefficients.csv"), row.names = FALSE)

perf <- as.data.frame(t(auc_ci(dat_model$pn_positive, dat_model$pred)))
perf$brier <- mean((dat_model$pn_positive - dat_model$pred)^2)
perf$n <- nrow(dat_model)
perf$events <- sum(dat_model$pn_positive)
perf$selected_variables <- paste(selected_base, collapse = "; ")
perf$lasso_cv_auc <- cv_auc
perf$lasso_cv_auc_se <- cv_auc_se
write.csv(perf, file.path(result_dir, "public_model_performance.csv"), row.names = FALSE)

vifs <- vif_table(dat_model, selected_base)
write.csv(vifs, file.path(result_dir, "public_model_vif.csv"), row.names = FALSE)

cal <- calibration_table(dat_model$pn_positive, dat_model$pred, groups = 5)
write.csv(cal, file.path(result_dir, "public_model_calibration.csv"), row.names = FALSE)

hl <- hosmer_lemeshow(dat_model$pn_positive, dat_model$pred, groups = 5)
write.csv(hl, file.path(result_dir, "public_model_hosmer_lemeshow.csv"), row.names = FALSE)

roc_obj <- pROC::roc(dat_model$pn_positive, dat_model$pred, quiet = TRUE, direction = "<")
best_threshold <- as.numeric(pROC::coords(roc_obj, x = "best", best.method = "youden", ret = "threshold"))
threshold_out <- threshold_metrics(dat_model$pn_positive, dat_model$pred, best_threshold)
write.csv(threshold_out, file.path(result_dir, "public_model_threshold_metrics.csv"), row.names = FALSE)

boot <- bootstrap_validate_auc(dat_model, final_formula, perf$auc, b = 200)
write.csv(boot, file.path(result_dir, "public_model_bootstrap_internal_validation.csv"), row.names = FALSE)

dca <- decision_curve(dat_model$pn_positive, dat_model$pred)
write.csv(dca, file.path(result_dir, "public_model_dca.csv"), row.names = FALSE)

saveRDS(final_fit, file.path(result_dir, "public_logistic_model.rds"))
write.csv(dat_model, file.path(result_dir, "public_model_dataset_with_predictions.csv"), row.names = FALSE)

png(file.path(result_dir, "public_model_roc.png"), width = 900, height = 700)
plot(roc_obj, main = "Public-compatible model ROC")
legend("bottomright", legend = sprintf("AUC %.3f", as.numeric(pROC::auc(roc_obj))), bty = "n")
dev.off()

png(file.path(result_dir, "public_model_calibration.png"), width = 900, height = 700)
plot(cal$pred, cal$obs, pch = 19, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Mean predicted risk", ylab = "Observed event rate", main = "Calibration by risk quintile")
abline(0, 1, lty = 2, col = "gray40")
text(cal$pred, cal$obs, labels = cal$group, pos = 3)
dev.off()

png(file.path(result_dir, "public_model_dca.png"), width = 900, height = 700)
plot(dca$threshold, dca$model, type = "l", lwd = 2, ylim = range(dca[, c("model", "treat_all", "treat_none")]),
     xlab = "Threshold probability", ylab = "Net benefit", main = "Decision curve")
lines(dca$threshold, dca$treat_all, lty = 2)
lines(dca$threshold, dca$treat_none, lty = 3)
legend("topright", legend = c("Model", "Treat all", "Treat none"), lty = c(1, 2, 3), bty = "n")
dev.off()

if ("rms" %in% rownames(installed.packages())) {
  try({
    dd <- rms::datadist(dat_model)
    options(datadist = "dd")
    lrm_fit <- rms::lrm(final_formula, data = dat_model, x = TRUE, y = TRUE)
    nom <- rms::nomogram(lrm_fit, fun = plogis, lp = FALSE,
                         fun.at = c(0.05, 0.1, 0.2, 0.3, 0.5),
                         funlabel = "pN+ risk")
    png(file.path(result_dir, "public_model_nomogram.png"), width = 1200, height = 850)
    plot(nom)
    dev.off()
  }, silent = TRUE)
}

write_stage_report(
  file.path(result_dir, "public_model_stage_report.md"),
  summary_lines, selected_base, perf, boot, threshold_out, hl, vifs, coef_table, cv_auc, cv_auc_se
)

cat("Selected variables:", paste(selected_base, collapse = ", "), "\n")
print(perf)
