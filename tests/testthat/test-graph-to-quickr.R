test_that("graph_to_quickr_function matches PJRT for scalar add", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x1, x2) {
      x1 + x2
    },
    list(
      x1 = nv_scalar(1.25, dtype = "f32"),
      x2 = nv_scalar(-0.5, dtype = "f32")
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  x1 <- 3.5
  x2 <- -2.25
  out_quick <- f_quick(x1, x2)
  out_pjrt <- eval_graph_pjrt(graph, x1, x2)
  expect_equal(out_quick, out_pjrt, tolerance = 1e-6)
})

test_that("graph_to_quickr_function matches PJRT for matmul", {
  testthat::skip_if_not_installed("quickr")

  X <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  B <- matrix(c(1, 0, 0, 1, 1, 1), nrow = 3, ncol = 2)

  graph <- trace_fn(
    function(X, B) {
      nv_matmul(X, B)
    },
    list(
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      B = nv_tensor(B, dtype = "f32", shape = dim(B))
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  out_quick <- f_quick(X, B)
  out_pjrt <- eval_graph_pjrt(graph, X, B)

  expect_equal(out_quick, out_pjrt, tolerance = 1e-5)
})

test_that("graph_to_quickr_function matches PJRT for batched matmul (rank-5)", {
  testthat::skip_if_not_installed("quickr")

  set.seed(3)

  b <- 2L
  t <- 3L
  h <- 2L
  m <- 4L
  n <- 3L
  p <- 5L

  A <- array(rnorm(b * t * h * m * n, sd = 0.2), dim = c(b, t, h, m, n))
  B <- array(rnorm(b * t * h * n * p, sd = 0.2), dim = c(b, t, h, n, p))

  graph <- trace_fn(
    function(A, B) {
      nv_matmul(A, B)
    },
    list(
      A = nv_tensor(A, dtype = "f32", shape = dim(A)),
      B = nv_tensor(B, dtype = "f32", shape = dim(B))
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  out_quick <- f_quick(A, B)
  out_pjrt <- eval_graph_pjrt(graph, A, B)

  expect_equal(dim(out_quick), c(b, t, h, m, p))
  expect_equal(out_quick, out_pjrt, tolerance = 1e-4)
})

test_that("graph_to_quickr_function matches PJRT for sum reduction", {
  testthat::skip_if_not_installed("quickr")

  x <- c(1, 2, 3, 4, 5)

  graph <- trace_fn(
    function(x) {
      sum(x)
    },
    list(x = nv_tensor(x, dtype = "f32", shape = c(length(x))))
  )

  f_quick <- graph_to_quickr_function(graph)

  out_quick <- f_quick(x)
  out_pjrt <- eval_graph_pjrt(graph, x)

  expect_equal(out_quick, out_pjrt, tolerance = 1e-6)
})

test_that("graph_to_quickr_function supports list outputs", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) {
      list(a = x, b = x + x)
    },
    list(x = nv_scalar(1.0, dtype = "f32"))
  )

  f_quick <- graph_to_quickr_function(graph)
  out_quick <- f_quick(0.5)
  out_pjrt <- eval_graph_pjrt(graph, 0.5)
  expect_equal(out_quick, out_pjrt, tolerance = 1e-6)
})

test_that("graph_to_quickr_function rejects nested inputs", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) {
      x$a + x$b
    },
    list(x = list(
      a = nv_scalar(1.0, dtype = "f32"),
      b = nv_scalar(2.0, dtype = "f32")
    ))
  )

  expect_error(graph_to_quickr_function(graph), "flat", fixed = FALSE)
})
