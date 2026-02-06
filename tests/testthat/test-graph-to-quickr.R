test_that("graph_to_quickr_function supports list outputs", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) {
      list(a = x, b = x + x)
    },
    list(x = nv_scalar(1.0, dtype = "f64"))
  )

  f_quick <- graph_to_quickr_function(graph)
  out_quick <- f_quick(0.5)
  out_pjrt <- eval_graph_pjrt(graph, 0.5)
  expect_equal(out_quick, out_pjrt)
})

test_that("graph_to_quickr_function supports nested inputs", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) {
      x$a + x$b
    },
    list(
      x = list(
        a = nv_scalar(1.0, dtype = "f64"),
        b = nv_scalar(2.0, dtype = "f64")
      )
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  out_quick <- f_quick(x = list(a = 0.5, b = 1.25))
  out_pjrt <- eval_graph_pjrt(graph, list(a = 0.5, b = 1.25))
  expect_equal(out_quick, out_pjrt)
})

test_that("graph_to_quickr_function handles GraphLiteral inputs (R scalar literals)", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) {
      x + 1
    },
    list(x = nv_scalar(0.0, dtype = "f64"))
  )

  f_quick <- graph_to_quickr_function(graph)
  out_quick <- f_quick(2.5)
  out_pjrt <- eval_graph_pjrt(graph, 2.5)
  expect_equal(out_quick, out_pjrt)
})

test_that("graph_to_quickr_function produces a stable flat signature", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(out, v1) {
      out + v1
    },
    list(
      out = nv_scalar(1.0, dtype = "f64"),
      v1 = nv_scalar(2.0, dtype = "f64")
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  expect_identical(names(formals(f_quick)), c("x1", "x2"))

  out_quick <- f_quick(0.5, 1.25)
  out_pjrt <- eval_graph_pjrt(graph, 0.5, 1.25)
  expect_equal(out_quick, out_pjrt)
})
