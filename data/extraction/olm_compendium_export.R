# Export the OpenLandMap SoilSamples horizon table (Zenodo 19731002, sol_chem.pnts_horizons.rds) to a CSV for
# geosoil.corpus.olm_compendium, keeping only the columns and source databases the loader uses. Base R only.
#   Rscript data/extraction/olm_compendium_export.R <in.rds> <out.csv.gz> <comma-separated source_db list>
args <- commandArgs(trailingOnly = TRUE)
h <- as.data.frame(readRDS(args[1]))
keep <- strsplit(args[3], ",")[[1]]
cols <- c("site_key", "source_db", "site_obsdate", "longitude_decimal_degrees", "latitude_decimal_degrees",
          "layer_key", "hzn_top", "hzn_bot", "clay_total", "sand_total", "organic_carbon", "ph_h2o", "ph_cacl2",
          "cec_nh4_ph_7", "sum_of_cations_cec_pH_8_2", "ca_ext", "mg_ext", "k_ext", "bulk_density_oven_dry")
h <- h[h$source_db %in% keep, cols]
write.csv(h, gzfile(args[2]), row.names = FALSE)
cat(nrow(h), "horizons from", length(unique(h$source_db)), "sources ->", args[2], "\n")
