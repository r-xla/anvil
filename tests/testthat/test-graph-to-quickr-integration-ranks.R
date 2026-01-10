test_that("integration: `%*%` matches PJRT for tensor ranks 1..5", {
  testthat::skip_if_not_installed("quickr")

  set.seed(4)

  m <- 3L
  n <- 4L
  p <- 5L
  batch_dim <- 2L

  matmul_any_rank <- function(A, B) {
    if (ndims(A) == 1L) {
      A <- nv_reshape(A, c(1L, shape(A)))
    }
    if (ndims(B) == 1L) {
      B <- nv_reshape(B, c(shape(B), 1L))
    }
    A %*% B
  }

  template_tensor <- function(x) {
    shp <- dim(x)
    if (is.null(shp)) {
      shp <- c(length(x))
    }
    nv_tensor(x, dtype = "f64", shape = shp)
  }

  make_input <- function(rank, side) {
    if (rank == 1L) {
      return(rnorm(n, sd = 0.2))
    }
    leading <- rep.int(batch_dim, rank - 2L)
    if (side == "lhs") {
      dims <- c(leading, m, n)
    } else {
      dims <- c(leading, n, p)
    }
    array(rnorm(prod(dims), sd = 0.2), dim = dims)
  }

  for (a_rank in 1:5) {
    for (b_rank in 1:5) {
      A <- make_input(a_rank, "lhs")
      B <- make_input(b_rank, "rhs")

      graph <- trace_fn(
        matmul_any_rank,
        list(
          A = template_tensor(A),
          B = template_tensor(B)
        )
      )

      f_quick <- graph_to_quickr_function(graph)
      run_pjrt <- compile_graph_pjrt(graph)

      out_quick <- f_quick(A, B)
      out_pjrt <- run_pjrt(A, B)

      testthat::expect_equal(
        out_quick,
        out_pjrt,
        tolerance = 1e-12,
        info = paste0("a_rank=", a_rank, ", b_rank=", b_rank)
      )
    }
  }
})
