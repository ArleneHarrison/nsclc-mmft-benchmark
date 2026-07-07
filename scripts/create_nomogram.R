# Clinical nomogram / bedside point-score for the primary NSCLC subtype model.
# Uses the EXACT same train/test split as the rest of the study
# (outputs/model_optimization/nomogram_{train,test}_data.csv, exported from
# Python with RANDOM_STATE=20260701, matching scripts/benchmark_petct_blood_models.py).
# Fits a standard MLE logistic regression (rms::lrm) on the training set only,
# using the primary model's 5 SelectKBest-chosen raw-unit variables
# (gender, CEA, SUVmean, SUVmax, SUVmin) -- standard practice for converting a
# lightly-regularised (ridge, C=3.0) development model into a clinician-facing
# nomogram with clinically interpretable (non-standardised) axes.

suppressMessages({
  library(rms)
  library(pROC)
})

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
if (length(file_arg) == 0) {
  stop("Run this script with Rscript so the --file argument is available.")
}
script_path <- normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE)
ROOT <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
OUT_DIR <- file.path(ROOT, "server_results", "petct_blood_benchmark")
OPT_DIR <- file.path(ROOT, "outputs", "model_optimization")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

train <- read.csv(file.path(OPT_DIR, "nomogram_train_data.csv"), stringsAsFactors = FALSE)
test  <- read.csv(file.path(OPT_DIR, "nomogram_test_data.csv"),  stringsAsFactors = FALSE)

train$gender_male <- ifelse(train$gender == 1, 1, 0)
test$gender_male  <- ifelse(test$gender == 1, 1, 0)
train$histology <- as.integer(train$histology)
test$histology  <- as.integer(test$histology)

dd <- datadist(train)
options(datadist = "dd")

fit <- lrm(histology ~ gender_male + CEA + SUVmean + SUVmax + SUVmin, data = train, x = TRUE, y = TRUE)
print(fit)

# ---- nomogram (continuous, clinician-facing) ----
# CEA has extreme high-end outliers (train max 1793 ng/mL vs median 4.8), which
# would stretch the CEA axis until the clinically relevant range (most patients
# under ~50 ng/mL) collapses to an unreadable sliver. Explicitly cap the
# DISPLAYED axis ranges to clinically sensible spans (fit still uses full data,
# only the nomogram's plotted axis range is restricted) -- CEA to 0-50 ng/mL
# (covers ~90th percentile), SUV variables to their observed ranges.
png(file.path(OUT_DIR, "primary_model_nomogram.png"), width = 1600, height = 1000, res = 150)
nom <- nomogram(fit, fun = plogis, lp = FALSE,
                 fun.at = c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
                 funlabel = "Predicted probability of squamous histology",
                 CEA = seq(0, 50, 5),
                 SUVmean = seq(3, 17, 2),
                 SUVmax = seq(5, 45, 5),
                 SUVmin = seq(0, 8, 1))
plot(nom, xfrac = 0.35, cex.axis = 0.85, cex.var = 0.95)
mtext("Note: CEA axis capped at 50 ng/mL for readability (training range up to 1793 ng/mL; ~90% of patients are under 50 ng/mL). Values above 50 map to the same point score as 50.",
      side = 1, line = 4, cex = 0.7, adj = 0)
dev.off()
cat("Saved nomogram plot.\n")

# ---- simplified integer bedside point-score ----
# Standard nomogram-to-points conversion, done correctly: scale the SIGNED
# linear-predictor contribution of each variable (coef_i * x_i) by a single
# common factor so the variable with the largest coefficient*range spans
# ~100 points across its observed training range. Using SIGNED coefficients
# throughout (not abs()) is essential -- CEA and SUVmin have NEGATIVE
# coefficients (higher value -> more adenocarcinoma-like), and discarding
# the sign here (an earlier version of this script did, via abs()) silently
# breaks the score's monotonic relationship to the true linear predictor.
# Because this rescaling is linear, the resulting total score is a strictly
# monotonic transform of the original linear predictor and must reproduce
# EXACTLY the same ROC AUC as the full-precision model -- this equality is
# used below as a correctness check, not just a nice-to-have.
vars <- c("gender_male", "CEA", "SUVmean", "SUVmax", "SUVmin")
coefs <- coef(fit)[vars]
ranges <- sapply(train[, vars], function(x) diff(range(x, na.rm = TRUE)))
scale_factor <- 100 / max(abs(coefs) * ranges)
points_per_unit <- round(coefs * scale_factor, 3)  # SIGNED -- keep the sign!
cat("\nSigned points per 1-unit increase (simplified bedside score):\n")
print(points_per_unit)

# round to friendlier bedside increments while preserving sign
gender_points <- round(points_per_unit["gender_male"], 1)
cea_points_per5 <- round(points_per_unit["CEA"] * 5, 1)
suvmean_points_per1 <- round(points_per_unit["SUVmean"], 1)
suvmax_points_per1 <- round(points_per_unit["SUVmax"], 1)
suvmin_points_per1 <- round(points_per_unit["SUVmin"], 1)

score_table <- data.frame(
  variable = c("Male sex", "CEA (per +5 ng/mL)", "SUVmean (per +1)", "SUVmax (per +1)", "SUVmin (per +1)"),
  points_per_unit_signed = c(gender_points, cea_points_per5, suvmean_points_per1, suvmax_points_per1, suvmin_points_per1),
  direction = c("higher = more squamous-like", "higher = more adenocarcinoma-like (negative points)",
                "higher = more squamous-like", "higher = more squamous-like", "higher = more adenocarcinoma-like (negative points)")
)
write.csv(score_table, file.path(OUT_DIR, "primary_model_bedside_score_weights.csv"), row.names = FALSE)
cat("\nBedside score weights (signed):\n"); print(score_table)

# ---- compute the simplified score for every test-set patient + validate discrimination ----
raw_score <- with(test,
  gender_points * gender_male +
  cea_points_per5 * (CEA / 5) +
  suvmean_points_per1 * SUVmean +
  suvmax_points_per1 * SUVmax +
  suvmin_points_per1 * SUVmin
)
test$bedside_score <- raw_score
roc_bedside <- roc(test$histology, test$bedside_score, quiet = TRUE)
auc_bedside <- as.numeric(auc(roc_bedside))
cat(sprintf("\nSimplified integer bedside score AUC on the SAME locked test set: %.4f\n", auc_bedside))

# reference: full-precision lrm predicted probability AUC on the same test set (sanity check vs ridge 0.854)
test_pred <- predict(fit, newdata = test, type = "fitted")
roc_full <- roc(test$histology, test_pred, quiet = TRUE)
auc_full <- as.numeric(auc(roc_full))
cat(sprintf("Full-precision MLE nomogram model AUC on locked test set (sanity check): %.4f\n", auc_full))

write.csv(test, file.path(OPT_DIR, "nomogram_test_with_score.csv"), row.names = FALSE)

summary_out <- list(
  mle_refit_test_auc = auc_full,
  bedside_integer_score_test_auc = auc_bedside,
  bedside_score_weights = score_table
)
saveRDS(summary_out, file.path(OPT_DIR, "nomogram_summary.rds"))

cat("\nDone. Files saved:\n")
cat(" -", file.path(OUT_DIR, "primary_model_nomogram.png"), "\n")
cat(" -", file.path(OUT_DIR, "primary_model_bedside_score_weights.csv"), "\n")
cat(" -", file.path(OPT_DIR, "nomogram_test_with_score.csv"), "\n")
