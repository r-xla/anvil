library(anvil)

nr <- 5000L
nc <- nr

nsteps <- 10L

X <- nv_tensor(matrix(rnorm(nr * nc), nr, nc))
Y <- nv_tensor(matrix(rnorm(nr * nc), nr, nc))

matmul <- jit(nv_matmul)
matmul(X, Y)

bench::mark(
  pjrt::value(matmul(X, Y)$tensor)
)


f <- function(sleep) {
  out <- X
  for (i in seq_len(nsteps)) {
    out <- matmul(out, Y)
    Sys.sleep(sleep)
  }
  as_array(out)
}

f(0.1)

result <- bench::mark(
  f(0.0),
  f(0.05),
  f(0.1),
  f(0.2),
  f(0.5),
  f(1),
  f(2),
  check = FALSE,
  iterations = 10
)

print(result)


bench::mark(
  Sys.sleep(0.1),
  Sys.sleep(0.2),
  check = FALSE
)
