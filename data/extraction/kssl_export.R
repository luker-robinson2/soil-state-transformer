# Export the KSSL Lab Data Mart sites + layers to two CSVs for geosoil.corpus.kssl.
#
# Reads either source of the same LDM tables:
#   - the official GeoPackage snapshot (ncss_labdata.gpkg, ~5.5 GB) from
#     https://ncsslabdatamart.sc.egov.usda.gov/database_download.aspx
#   - the monthly labtaxa mirror (cached-LDM-SPC.rds, ~64 MB) from
#     https://github.com/brownag/labtaxa/releases  (used while the USDA site is down)
#
# Usage:
#   Rscript data/extraction/kssl_export.R data/raw/kssl/cached-LDM-SPC.rds data/raw/kssl
#   Rscript data/extraction/kssl_export.R data/raw/kssl/ncss_labdata.gpkg  data/raw/kssl
# The .rds path needs only base R; the .gpkg path needs DBI + RSQLite.

args <- commandArgs(trailingOnly = TRUE)
src <- args[1]
out <- if (length(args) > 1) args[2] else dirname(src)

site_cols <- c("pedon_key", "site_key", "pedlabsampnum", "country_key", "site_obsdate",
               "latitude_std_decimal_degrees", "longitude_std_decimal_degrees",
               "latitude_decimal_degrees", "longitude_decimal_degrees")
layer_cols <- c("pedon_key", "layer_key", "labsampnum", "hzn_top", "hzn_bot", "prep_code",
                "sand_total", "silt_total", "clay_total", "particle_size_method",
                "total_carbon_ncs", "total_carbon_ncs_method", "caco3_lt_2_mm",
                "organic_carbon_walkley_black", "oc_walkley_black_method",
                "ph_h2o", "ph_h2o_method", "ph_cacl2", "ph_cacl2_method",
                "bulk_density_third_bar", "bulk_density_third_bar_method",
                "bulk_density_oven_dry", "bulk_density_oven_dry_method")

if (grepl("\\.rds$", src)) {
  # A SoilProfileCollection; read its slots as attributes so aqp isn't required.
  a <- attributes(readRDS(src))
  site <- a$site
  layer <- a$horizons
} else {
  con <- DBI::dbConnect(RSQLite::SQLite(), src)
  site <- DBI::dbGetQuery(con, "
    SELECT c.*, s.latitude_std_decimal_degrees, s.longitude_std_decimal_degrees
    FROM lab_combine_nasis_ncss c LEFT JOIN lab_site s ON c.site_key = s.site_key")
  layer <- DBI::dbGetQuery(con, "
    SELECT l.*, p.*, c.*
    FROM lab_layer l
    LEFT JOIN lab_physical_properties p ON l.labsampnum = p.labsampnum
    LEFT JOIN lab_chemical_properties c ON l.labsampnum = c.labsampnum")
  layer <- layer[, !duplicated(names(layer))]
  DBI::dbDisconnect(con)
}

missing <- c(setdiff(site_cols, names(site)), setdiff(layer_cols, names(layer)))
if (length(missing)) stop("missing LDM columns: ", paste(missing, collapse = ", "))

site <- as.data.frame(site)[, site_cols]
layer <- as.data.frame(layer)[, layer_cols]
write.csv(site, gzfile(file.path(out, "kssl_sites.csv.gz")), row.names = FALSE)
write.csv(layer, gzfile(file.path(out, "kssl_layers.csv.gz")), row.names = FALSE)
cat(nrow(site), "sites,", nrow(layer), "layers ->", out, "\n")
