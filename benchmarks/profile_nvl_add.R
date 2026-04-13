suppressPackageStartupMessages(devtools::load_all("/Users/sebi/r-xla/.worktrees/anvil"))

# Warm up - ensure plugin loaded, devices ready, etc.
x0 <- nv_array(1:3)
y0 <- nv_array(4:6)
invisible(nvl_add(x0, y0))

mk <- function(shape) {
  list(
    nv_array(array(runif(prod(shape)), shape), dtype = "f32"),
    nv_array(array(runif(prod(shape)), shape), dtype = "f32")
  )
}

# Each call with a FRESH shape forces a cache miss since shape is part of the cache key.
shapes <- lapply(seq_len(60), function(i) c(i + 10L, 3L))
pairs <- lapply(shapes, mk)

cat("=== Timing cold vs warm path ===\n")
t_cold <- sapply(pairs, function(p) {
  system.time(nvl_add(p[[1]], p[[2]]))[["elapsed"]]
})
cat(sprintf("Cold (first): %.4fs\n", t_cold[1]))
cat(sprintf("Cold (median of 60 fresh shapes): %.4fs\n", median(t_cold)))
cat(sprintf("Cold (mean of 60): %.4fs\n", mean(t_cold)))

# Warm (repeat, same shape -> cache hit)
p <- mk(c(32L, 32L))
nvl_add(p[[1]], p[[2]]) # prime
t_warm <- replicate(50, system.time(nvl_add(p[[1]], p[[2]]))[["elapsed"]])
cat(sprintf("Warm (median of 50): %.4fs\n", median(t_warm)))

cat("\n=== Rprof on 80 cold calls ===\n")
pairs2 <- lapply(lapply(seq_len(80), function(i) c(i + 100L, 5L)), mk)
prof_file <- tempfile(fileext = ".Rprof")
Rprof(prof_file, interval = 0.002, line.profiling = TRUE)
for (p in pairs2) {
  nvl_add(p[[1]], p[[2]])
}
Rprof(NULL)
s <- summaryRprof(prof_file)
cat("\n--- by.total (top 30) ---\n")
print(head(s$by.total, 30))
cat("\n--- by.self (top 25) ---\n")
print(head(s$by.self, 25))
cat(sprintf("\nTotal sample time: %.3fs\n", s$sampling.time))
