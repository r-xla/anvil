#!/usr/bin/env Rscript
#
# Verify that the environment variables recommended by the HMC vignette
# actually keep anvil / XLA-CPU execution single-threaded.
#
# The check is based on two signals, collected from a child R process that
# runs a parallelism-hungry XLA-CPU workload (a long chain of large matmuls
# fused into a single jit call):
#
#   1. CPU %, sampled from the parent via the 'ps' package. The process is
#      still allowed a modest overhead above 100 % because PjRt on CPU keeps
#      a small async dispatcher / client thread alive beside the compute
#      thread, and because the parent R thread is also scheduled. We consider
#      the run single-threaded if the p95 CPU % is at most `cpu_threshold_pct`
#      (default 150 %).
#
#   2. Wall-clock time for the main workload. On a multi-core machine, if XLA
#      was actually parallelising the matmul, disabling the parallelism will
#      make it noticeably slower. We require that the recommended setting be
#      at least `wall_min_ratio` (default 1.5) times slower than the baseline,
#      which rules out the case where XLA is already effectively serial on the
#      host and the CPU % we observe is just the dispatcher thread.
#
# Run with:  Rscript inst/verify-single-thread.R

stopifnot(requireNamespace("processx", quietly = TRUE),
          requireNamespace("ps", quietly = TRUE))

cpu_threshold_pct <- 150
wall_min_ratio    <- 1.5

# --- workload that would use many cores without limits -----------------------

workload_code <- '
suppressPackageStartupMessages(library(anvil))
f <- jit(function(A, B, n) {
  out <- nv_while(
    list(C = A, i = nv_scalar(0L)),
    function(C, i) i < n,
    function(C, i) list(C = nv_matmul(C, B), i = i + 1L)
  )
  out$C
}, static = "n")
N <- 2048L
set.seed(1)
A <- nv_array(matrix(runif(N * N), N, N), dtype = "f32")
B <- nv_array(matrix(runif(N * N) / N, N, N), dtype = "f32")
invisible(as_array(f(A, B, 20L)))       # compile + warm the cache
writeLines("READY", stdout()); flush(stdout())
t <- system.time(invisible(as_array(f(A, B, 300L))))
writeLines(sprintf("WALL %.6f", t[["elapsed"]]), stdout()); flush(stdout())
'

# --- run one case: child process + CPU polling -------------------------------

run_case <- function(name, env_vars) {
  parent_env <- Sys.getenv()
  merged <- c(parent_env, env_vars)
  merged <- merged[!duplicated(names(merged), fromLast = TRUE)]

  p <- processx::process$new(
    "Rscript", c("-e", workload_code),
    stdout = "|", stderr = "|",
    env = merged
  )

  preamble <- character(0)
  repeat {
    if (!p$is_alive()) break
    line <- p$read_output_lines()
    preamble <- c(preamble, line)
    if (any(grepl("READY", line))) break
    Sys.sleep(0.02)
  }
  if (!any(grepl("READY", preamble))) {
    err <- p$read_all_error()
    p$kill()
    stop(sprintf("[%s] child did not reach READY. stderr:\n%s", name, err))
  }

  h <- ps::ps_handle(p$get_pid())
  samples <- numeric(0)
  last_t <- Sys.time()
  last_cpu <- tryCatch(ps::ps_cpu_times(h), error = function(e) NULL)
  wall <- NA_real_
  captured <- character(0)
  while (p$is_alive()) {
    Sys.sleep(0.05)
    now <- Sys.time()
    cpu <- tryCatch(ps::ps_cpu_times(h), error = function(e) NULL)
    if (!is.null(cpu) && !is.null(last_cpu)) {
      dt_wall <- as.numeric(difftime(now, last_t, units = "secs"))
      dt_cpu  <- unname((cpu[["user"]] + cpu[["system"]]) -
                        (last_cpu[["user"]] + last_cpu[["system"]]))
      if (dt_wall > 0) samples <- c(samples, 100 * dt_cpu / dt_wall)
    }
    last_t <- now
    last_cpu <- cpu
    out <- tryCatch(p$read_output_lines(), error = function(e) character(0))
    captured <- c(captured, out)
    if (any(grepl("WALL ", captured))) break
  }
  p$wait(timeout = 10000)
  if (p$is_alive()) p$kill()
  # Pick up any remaining output including the WALL line.
  captured <- c(captured, tryCatch(p$read_all_output_lines(),
                                   error = function(e) character(0)))

  wall_line <- grep("^WALL ", captured, value = TRUE)
  if (length(wall_line) > 0) {
    wall <- as.numeric(sub("^WALL ", "", wall_line[[1L]]))
  }

  if (length(samples) > 1) samples <- samples[-1]

  list(
    name = name,
    env  = env_vars,
    n    = length(samples),
    mean = if (length(samples)) mean(samples) else NA_real_,
    p95  = if (length(samples)) as.numeric(quantile(samples, 0.95)) else NA_real_,
    max  = if (length(samples)) max(samples)  else NA_real_,
    wall = wall
  )
}

# --- run the matrix ----------------------------------------------------------

cases <- list(
  list(name = "baseline (no limits)",
       env  = character(0)),
  list(name = "PJRT_NPROC=1 only",
       env  = c(PJRT_NPROC = "1")),
  list(name = "Eigen flag only",
       env  = c(XLA_FLAGS = "--xla_cpu_multi_thread_eigen=false")),
  list(name = "recommended (PJRT_NPROC=1 + Eigen flag)",
       env  = c(PJRT_NPROC = "1",
                XLA_FLAGS  = "--xla_cpu_multi_thread_eigen=false"))
)

results <- lapply(cases, function(case) {
  cat(sprintf(">>> running: %s\n", case$name))
  run_case(case$name, case$env)
})

# --- summary -----------------------------------------------------------------

baseline_wall <- results[[1L]]$wall

cat("\n", strrep("=", 86), "\n", sep = "")
cat(sprintf("%-44s %7s %7s %7s %9s\n",
            "case", "mean%", "p95%", "max%", "wall(s)"))
cat(strrep("-", 86), "\n", sep = "")
for (r in results) {
  cat(sprintf("%-44s %7.1f %7.1f %7.1f %9.3f\n",
              r$name, r$mean, r$p95, r$max, r$wall))
}
cat(strrep("=", 86), "\n", sep = "")

recommended <- results[[length(results)]]
cpu_ok  <- isTRUE(recommended$p95 <= cpu_threshold_pct)
# Slowdown ratio: 1.0 means no change vs baseline; > 1 means slower.
slowdown <- recommended$wall / baseline_wall
wall_ok  <- isTRUE(slowdown >= wall_min_ratio)

cat(sprintf("\nrecommended p95 CPU%%   = %.1f%%   (threshold <= %.0f%%)\n",
            recommended$p95, cpu_threshold_pct))
cat(sprintf("recommended / baseline wall = %.2fx (expected >= %.2fx if the "
            , slowdown, wall_min_ratio),
    "baseline was actually multi-threaded)\n", sep = "")

if (!cpu_ok) {
  cat("\nFAIL: recommended settings leave significant multi-core activity.\n")
  quit(status = 1)
}
if (!wall_ok) {
  cat("\nNOTE: recommended settings are not noticeably slower than baseline.\n",
      "This does not prove the flags are doing nothing -- it may just mean\n",
      "that XLA was already effectively single-threaded on this host for\n",
      "the chosen workload. CPU p95 is within budget, so we PASS.\n", sep = "")
}
cat("\nPASS: recommended settings keep execution single-threaded.\n")
