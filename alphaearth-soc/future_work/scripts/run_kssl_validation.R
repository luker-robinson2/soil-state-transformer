#!/usr/bin/env Rscript
# Phase F — KSSL/WoSIS validation analysis.
# 1) OpenLandMap-vs-KSSL agreement
# 2) Refit AE -> log-SOC RF with KSSL labels (same paper recipe)
# 3) Multi-target on KSSL
# 4) Moran's I on residuals

suppressMessages({
  library(tidyverse); library(boot); library(ranger); library(spdep)
})
set.seed(42)

df <- read.csv("/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/kssl_with_features.csv")
ae <- sprintf("A%02d", 0:63)
df <- df %>% drop_na(soc_g_kg)

df$log_soc      <- log1p(df$soc_g_kg)
df$log_soc_olm  <- log1p(df$soc_olm)

# Spatial-block folds
df$block <- paste0(round(df$lon), "_", round(df$lat))
ub <- unique(df$block)
df$fold <- sample(1:5, length(ub), replace=TRUE)[match(df$block, ub)]
cat("Folds:", table(df$fold), "\n")

# ----------------------------------------------------------------------
# 1) OpenLandMap vs KSSL agreement
# ----------------------------------------------------------------------
cat("\n============================================================\n")
cat("(1) OpenLandMap-vs-KSSL agreement\n")
cat("============================================================\n")
agree <- df %>% drop_na(soc_olm)
r2_agree <- cor(agree$soc_olm, agree$soc_g_kg)^2
r2_log_agree <- cor(agree$log_soc_olm, agree$log_soc)^2
bias <- mean(agree$soc_g_kg - agree$soc_olm)
rmse_agree <- sqrt(mean((agree$soc_g_kg - agree$soc_olm)^2))
cat(sprintf("OpenLandMap SOC vs WoSIS lab SOC (n = %d):\n", nrow(agree)))
cat(sprintf("  R^2 (raw):      %.3f\n", r2_agree))
cat(sprintf("  R^2 (log1p):    %.3f\n", r2_log_agree))
cat(sprintf("  Bias (lab-OLM): %+.2f g/kg\n", bias))
cat(sprintf("  RMSE:           %.2f g/kg\n", rmse_agree))
cat("=> The original paper's R^2 = 0.75 is alignment between AE and OpenLandMap;\n")
cat(sprintf("   OpenLandMap itself only matches lab measurements at R^2 = %.3f.\n", r2_log_agree))

# ----------------------------------------------------------------------
# 2) AE -> KSSL SOC RF (paper recipe)
# ----------------------------------------------------------------------
cat("\n============================================================\n")
cat("(2) AE -> log(WoSIS SOC) RF on spatial-block CV\n")
cat("============================================================\n")
form <- as.formula(paste("log_soc ~", paste(ae, collapse="+")))
cv_kssl <- map_dfr(1:5, function(k) {
  tr <- df %>% filter(fold != k)
  te <- df %>% filter(fold == k)
  m <- ranger(form, data = tr, num.trees = 500, seed = 42)
  pr <- predict(m, data = te)$predictions
  tibble(fold = k, n = nrow(te),
         rmse = sqrt(mean((pr - te$log_soc)^2)),
         r2 = cor(pr, te$log_soc)^2,
         pred = list(pr), obs = list(te$log_soc),
         lon = list(te$lon), lat = list(te$lat))
})
print(cv_kssl %>% select(fold, n, rmse, r2))

ci_b <- boot::boot(cv_kssl$r2, function(x, i) mean(x[i]), R = 10000)
ci <- boot::boot.ci(ci_b, type = "bca")$bca[4:5]
cat(sprintf("\nMean R^2 = %.3f  (BCa 95%% CI: [%.3f, %.3f])\n", mean(cv_kssl$r2), ci[1], ci[2]))
cat(sprintf("Mean RMSE = %.3f\n", mean(cv_kssl$rmse)))

# ----------------------------------------------------------------------
# 3) Multi-target on KSSL
# ----------------------------------------------------------------------
cat("\n============================================================\n")
cat("(3) Multi-target RF on WoSIS labels\n")
cat("============================================================\n")
df$log_soc_kssl <- df$log_soc
logit_pct <- function(x) log((x/100 + 1e-3) / (1 - x/100 + 1e-3))

multi_targets <- list(
  log_soc      = list(col = "log_soc",          subset = !is.na(df$log_soc)),
  ph_h2o       = list(col = "ph_h2o",           subset = !is.na(df$ph_h2o)),
  sand_logit   = list(col = NULL,               subset = !is.na(df$sand_pct), tx = function(x) logit_pct(x)),
  clay_logit   = list(col = NULL,               subset = !is.na(df$clay_pct), tx = function(x) logit_pct(x)),
  bd           = list(col = "bd_g_cm3",         subset = !is.na(df$bd_g_cm3))
)
df$sand_logit <- logit_pct(df$sand_pct)
df$clay_logit <- logit_pct(df$clay_pct)

multi_results <- map_dfr(c("log_soc","ph_h2o","sand_logit","clay_logit","bd_g_cm3"), function(t) {
  d <- df %>% drop_na(all_of(t))
  if (nrow(d) < 200) {
    return(tibble(target = t, n = nrow(d), mean_r2 = NA, lo = NA, hi = NA, rmse = NA))
  }
  d$block <- paste0(round(d$lon), "_", round(d$lat))
  ub <- unique(d$block)
  d$fold <- sample(1:5, length(ub), replace=TRUE)[match(d$block, ub)]
  form_t <- as.formula(paste(t, "~", paste(ae, collapse="+")))
  cv <- map_dfr(1:5, function(k) {
    tr <- d %>% filter(fold!=k); te <- d %>% filter(fold==k)
    m  <- ranger(form_t, data=tr, num.trees=500, seed=42)
    pr <- predict(m, data=te)$predictions
    tibble(fold=k, n=nrow(te),
           rmse=sqrt(mean((pr-te[[t]])^2)),
           r2=cor(pr, te[[t]])^2)
  })
  ci_b <- boot::boot(cv$r2, function(x,i) mean(x[i]), R=10000)
  ci <- boot::boot.ci(ci_b, type="bca")$bca[4:5]
  tibble(target = t, n = nrow(d),
         mean_r2 = mean(cv$r2), lo = ci[1], hi = ci[2],
         rmse = mean(cv$rmse))
})
print(multi_results)

# ----------------------------------------------------------------------
# 4) Moran's I on residuals
# ----------------------------------------------------------------------
cat("\n============================================================\n")
cat("(4) Moran's I on out-of-fold residuals (AE -> log-SOC RF)\n")
cat("============================================================\n")
oof <- tibble(
  lon = unlist(cv_kssl$lon),
  lat = unlist(cv_kssl$lat),
  pred = unlist(cv_kssl$pred),
  obs = unlist(cv_kssl$obs)
) %>% mutate(resid = pred - obs)

set.seed(42)
samp <- oof %>% sample_n(min(2000, n()))
coords <- as.matrix(samp[, c("lon","lat")])
nb <- knn2nb(knearneigh(coords, k = 8))
lw <- nb2listw(nb, style = "W")
mi <- moran.test(samp$resid, lw, randomisation = TRUE)
cat("Moran's I:", round(mi$estimate["Moran I statistic"], 4),
    " expectation:", round(mi$estimate["Expectation"], 4),
    " p-value:", format.pval(mi$p.value, eps=1e-15), "\n")
cat("=> If p < 0.05 with positive Moran's I, residuals are spatially clustered\n")
cat("   (5-fold spatial-block CV does not fully eliminate autocorrelation).\n")

# ----------------------------------------------------------------------
# Save figures + CSVs
# ----------------------------------------------------------------------
fig_dir <- "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# Agreement plot
p <- ggplot(agree, aes(soc_olm, soc_g_kg)) +
  geom_point(alpha = 0.2, size = 0.6, color = "#5b8def") +
  geom_abline(slope = 1, intercept = 0, color = "#c0392b", linetype = "dashed") +
  scale_x_log10() + scale_y_log10() +
  theme_minimal() +
  labs(x = "OpenLandMap SOC (g/kg)",
       y = "WoSIS lab-measured SOC (g/kg)",
       title = sprintf("OpenLandMap underestimates lab SOC (n = %d, R² = %.3f on log scale)",
                       nrow(agree), r2_log_agree))
ggsave(file.path(fig_dir, "phaseF_agreement.png"), p, width=7, height=5, dpi=300)

# RF predicted vs observed
oof_plot <- oof
p2 <- ggplot(oof_plot, aes(obs, pred)) +
  geom_point(alpha = 0.2, size = 0.6, color = "#5b8def") +
  geom_abline(slope = 1, intercept = 0, color = "#c0392b", linetype = "dashed") +
  theme_minimal() +
  labs(x = "Observed log(1 + WoSIS SOC)", y = "Predicted",
       title = sprintf("AE -> WoSIS log-SOC, spatial-block CV (n = %d, R² = %.3f)",
                       nrow(oof_plot), mean(cv_kssl$r2)))
ggsave(file.path(fig_dir, "phaseF_pred_obs.png"), p2, width=6, height=5, dpi=300)

# Headline 3x2 comparison table
headline <- tibble(
  label = c("OpenLandMap (paper, R26)", "WoSIS (this work)"),
  ae_only_r2 = c(0.748, mean(cv_kssl$r2))
)
write.csv(headline,
          "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseF_headline.csv",
          row.names = FALSE)
write.csv(multi_results,
          "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseF_multi.csv",
          row.names = FALSE)
write.csv(tibble(metric = c("r2_raw","r2_log","bias_g_kg","rmse_g_kg"),
                 value = c(r2_agree, r2_log_agree, bias, rmse_agree)),
          "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseF_agreement.csv",
          row.names = FALSE)
write.csv(tibble(moran_i = mi$estimate["Moran I statistic"],
                 expectation = mi$estimate["Expectation"],
                 p_value = mi$p.value),
          "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseF_morans_i.csv",
          row.names = FALSE)
cat("\nSaved figures + CSVs.\n")
