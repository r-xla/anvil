library(anvil)

sizes <- c(1L, 2L, 4L, 8L, 16L, 32L, 64L, 128L, 256L, 512L, 1024L, 2048L)

identity_jit <- jit(function(x) x)
matmul_jit <- jit(nv_matmul)

make_pair <- function(n) {
  a <- nv_tensor(matrix(rnorm(n * n), n, n))
  b <- nv_tensor(matrix(rnorm(n * n), n, n))
  list(a = a, b = b)
}

pairs <- lapply(sizes, make_pair)

# warm up all cache entries
for (i in seq_along(sizes)) {
  identity_jit(pairs[[i]]$a)
  matmul_jit(pairs[[i]]$a, pairs[[i]]$b)
}

overhead_bm <- bench::mark(
  as_array(identity_jit(pairs[[1]]$a)),
  iterations = 200,
  check = FALSE
)

matmul_bms <- lapply(seq_along(sizes), function(i) {
  n <- sizes[i]
  bm <- bench::mark(
    as_array(matmul_jit(pairs[[i]]$a, pairs[[i]]$b)),
    iterations = if (n <= 64) 200 else if (n <= 512) 50 else 10,
    check = FALSE
  )
  bm$n <- n
  bm
})

matmul_results <- do.call(rbind, matmul_bms)

cat("=== JIT call overhead (identity, cache hit) ===\n")
print(overhead_bm[, c("median", "min", "mem_alloc", "n_itr")])

cat("\n=== matmul (n,n) x (n,n) ===\n")
print(matmul_results[, c("n", "median", "min", "mem_alloc", "n_itr")])
