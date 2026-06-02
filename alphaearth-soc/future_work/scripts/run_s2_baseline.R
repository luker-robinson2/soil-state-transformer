#!/usr/bin/env Rscript
# Phase C — Sentinel-2 baseline benchmark.
# Three RFs on identical 5-fold spatial-block CV: S2 (10 feat),
# AE (64 feat), S2+AE (74 feat). Paired bootstrap on Delta-R^2.

suppressMessages({
  library(tidyverse); library(boot); library(ranger)
})
set.seed(42)

unified <- read.csv("/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/soil_unified.csv")
s2 <- read.csv("/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/s2_features.csv")

ae <- sprintf("A%02d", 0:63)
s2_features <- setdiff(colnames(s2), c("lon","lat"))
cat("S2 features:", paste(s2_features, collapse=", "), "\n")
cat("Unified rows:", nrow(unified), "  S2 rows (raw):", nrow(s2), "\n")

# Inner join on (lon,lat) — only points with both AE and S2.
df <- unified %>% inner_join(s2, by = c("lon", "lat"))
cat("After join:", nrow(df), " rows\n")
df <- df %>% drop_na(all_of(s2_features))
cat("After dropping NA S2 features:", nrow(df), " rows\n")

df$log_soc <- log1p(df$soc)

# Spatial-block folds
df$block <- paste0(round(df$lon), "_", round(df$lat))
ub <- unique(df$block)
df$fold <- sample(1:5, length(ub), replace=TRUE)[match(df$block, ub)]
cat("Folds:", table(df$fold), "\n")

rf_cv <- function(features, target) {
  form <- as.formula(paste(target, "~", paste(features, collapse="+")))
  map_dfr(1:5, function(k) {
    tr <- df %>% filter(fold != k)
    te <- df %>% filter(fold == k)
    m <- ranger(form, data = tr, num.trees = 500, seed = 42)
    pr <- predict(m, data = te)$predictions
    tibble(fold = k, n = nrow(te),
           rmse = sqrt(mean((pr - te[[target]])^2)),
           r2 = cor(pr, te[[target]])^2,
           pred = list(pr), obs = list(te[[target]]))
  })
}

cat("\n============================================================\n")
cat("Three RFs on identical 5-fold spatial-block CV (n =", nrow(df), ")\n")
cat("============================================================\n")
cv_s2   <- rf_cv(s2_features,        "log_soc")
cv_ae   <- rf_cv(ae,                 "log_soc")
cv_both <- rf_cv(c(s2_features, ae), "log_soc")

bench <- tibble(
  model     = c("S2 only","AE only","S2 + AE"),
  n_features = c(length(s2_features), length(ae), length(c(s2_features, ae))),
  mean_r2   = c(mean(cv_s2$r2),  mean(cv_ae$r2),  mean(cv_both$r2)),
  mean_rmse = c(mean(cv_s2$rmse),mean(cv_ae$rmse),mean(cv_both$rmse))
)
print(bench)

# Bootstrap CI on each mean R^2 (per-fold resample)
boot_mean <- function(x, B = 10000) {
  b <- boot::boot(x, function(v, i) mean(v[i]), R = B)
  ci <- boot::boot.ci(b, type = "bca")$bca[4:5]
  list(mean = mean(x), lo = ci[1], hi = ci[2])
}
ci_s2   <- boot_mean(cv_s2$r2)
ci_ae   <- boot_mean(cv_ae$r2)
ci_both <- boot_mean(cv_both$r2)
cat("\n95% BCa CIs on mean R^2:\n")
cat(sprintf("  S2 only:  %.3f  [%.3f, %.3f]\n", ci_s2$mean,   ci_s2$lo,   ci_s2$hi))
cat(sprintf("  AE only:  %.3f  [%.3f, %.3f]\n", ci_ae$mean,   ci_ae$lo,   ci_ae$hi))
cat(sprintf("  S2 + AE:  %.3f  [%.3f, %.3f]\n", ci_both$mean, ci_both$lo, ci_both$hi))

# Paired bootstrap on Delta R^2 across folds
cat("\n============================================================\n")
cat("Paired bootstrap on Delta-R^2 (per-fold resample)\n")
cat("============================================================\n")
fold_r2 <- tibble(fold=cv_s2$fold, r2_s2=cv_s2$r2, r2_ae=cv_ae$r2, r2_both=cv_both$r2)
print(fold_r2)

paired_boot <- function(a, b, label, B = 10000) {
  d <- a - b
  bo <- boot::boot(d, function(x, i) mean(x[i]), R = B)
  ci <- boot::boot.ci(bo, type = "bca")$bca[4:5]
  cat(sprintf("  %s:  delta = %+.4f  [%+.4f, %+.4f]\n", label, mean(d), ci[1], ci[2]))
}
paired_boot(fold_r2$r2_ae,   fold_r2$r2_s2,   "AE - S2     ")
paired_boot(fold_r2$r2_both, fold_r2$r2_ae,   "S2+AE - AE  ")
paired_boot(fold_r2$r2_both, fold_r2$r2_s2,   "S2+AE - S2  ")

# Top features per source
ae_r <- sapply(ae, function(c) cor(df[[c]], df$log_soc))
s2_r <- sapply(s2_features, function(c) cor(df[[c]], df$log_soc))
top_compare <- bind_rows(
  tibble(source = "AE", feature = names(ae_r), r = ae_r),
  tibble(source = "S2", feature = names(s2_r), r = s2_r)
) %>% group_by(source) %>% slice_max(abs(r), n = 5) %>% ungroup()
cat("\nTop 5 |r| per source vs log-SOC:\n")
print(top_compare)

# Save figures + CSVs
fig_dir <- "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

plot_df <- tibble(
  model = factor(c("S2 only","AE only","S2 + AE"),
                 levels = c("S2 only","AE only","S2 + AE")),
  r2 = c(ci_s2$mean, ci_ae$mean, ci_both$mean),
  lo = c(ci_s2$lo,   ci_ae$lo,   ci_both$lo),
  hi = c(ci_s2$hi,   ci_ae$hi,   ci_both$hi)
)
p <- ggplot(plot_df, aes(model, r2, fill = model)) +
  geom_col(width = 0.6) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2) +
  scale_fill_manual(values = c("#27ae60","#5b8def","#34495e"), guide = "none") +
  theme_minimal(base_size = 12) +
  labs(x = NULL, y = expression(R^2),
       title = "S2 vs AlphaEarth log-SOC R² on identical spatial-block CV",
       subtitle = sprintf("n = %d points; error bars = BCa 95%% CI from per-fold bootstrap", nrow(df)))
ggsave(file.path(fig_dir, "phaseC_s2_ae_bench.png"),
       p, width = 7, height = 5, dpi = 300)

write.csv(bench, "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseC_bench.csv", row.names=FALSE)
write.csv(fold_r2, "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseC_fold_r2.csv", row.names=FALSE)
write.csv(top_compare, "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseC_top_features.csv", row.names=FALSE)
cat("\nSaved figures + CSVs.\n")
