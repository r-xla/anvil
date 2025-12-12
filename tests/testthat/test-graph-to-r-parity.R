test_that("graph_to_r_function matches PJRT for random scalar graphs", {
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  set.seed(1)

  fs <- list(
    function(x, y) x + y,
    function(x, y) x * y + x,
    function(x, y) nv_tanh(x) + nv_exp(y),
    function(x, y) nv_log(x + nv_scalar(1.0, dtype = "f32")) - y,
    function(x, y) nv_max(x, y) - nv_min(x, y)
  )

  for (i in seq_len(5L)) {
    f <- fs[[sample.int(length(fs), 1L)]]
    x <- runif(1L)
    y <- runif(1L)

    graph <- trace_fn(
      f,
      list(
        x = nv_scalar(x, dtype = "f32"),
        y = nv_scalar(y, dtype = "f32")
      )
    )

    f_r <- graph_to_r_function(graph)
    got_r <- f_r(x, y)
    got_pjrt <- eval_graph_pjrt(graph, x, y)

    expect_equal(as.numeric(got_r), as.numeric(got_pjrt), tolerance = 1e-5)
  }
})

