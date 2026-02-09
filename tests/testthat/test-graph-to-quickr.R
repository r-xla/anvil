test_that("graph_to_quickr_function supports list outputs", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      list(a = x, b = x + x)
    },
    list(x = nv_scalar(1.0, dtype = "f64"))
  )

  expect_quickr_matches_pjrt(graph, 0.5)
})

test_that("graph_to_quickr_function rejects non-graph inputs", {
  expect_error(graph_to_quickr_function(1), "must be a .*AnvilGraph", fixed = FALSE)
})

test_that("graph_to_quickr_r_function rejects non-graph inputs", {
  expect_error(graph_to_quickr_r_function(1), "must be a .*AnvilGraph", fixed = FALSE)
})

test_that("graph_to_quickr_function supports nested inputs", {
  skip_if_no_quickr_or_pjrt()

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

  expect_quickr_matches_pjrt(graph, x = list(a = 0.5, b = 1.25))
})

test_that("graph_to_quickr_function handles GraphLiteral inputs (R scalar literals)", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      x + 1
    },
    list(x = nv_scalar(0.0, dtype = "f64"))
  )

  expect_quickr_matches_pjrt(graph, 2.5)
})

test_that("graph_to_quickr_function produces a stable flat signature", {
  skip_if_no_quickr_or_pjrt()

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
  f_r <- graph_to_quickr_r_function(graph)
  expect_identical(names(formals(f_quick)), c("x1", "x2"))
  expect_identical(names(formals(f_r)), c("x1", "x2"))

  expect_quickr_matches_pjrt(graph, 0.5, 1.25)
})

test_that("graph_to_quickr_function handles zero-length output leaves when packing", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function() {
      list(
        b = nv_fill(2.0, shape = integer(), dtype = "f64"),
        a = nv_fill(1.0, shape = c(0L, 2L), dtype = "f64"),
        c = nv_fill(3.0, shape = integer(), dtype = "f64")
      )
    },
    list()
  )

  out <- expect_quickr_matches_pjrt(graph)
  out_quick <- out$out_quick
  expect_identical(dim(out_quick$a), c(0L, 2L))
})

test_that("graph_to_quickr_function ignores static args when flattening nested inputs", {
  skip_if_no_quickr_or_pjrt()

  # Regression test: when tracing with nested list inputs and an extra static arg,
  # graph$inputs contains only the tensor leaves but wrapper flattening sees both.
  graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x$a$u + x$b
      } else {
        x$a$v + x$b
      }
    },
    list(
      x = list(
        a = list(
          u = nv_scalar(1.0, dtype = "f64"),
          v = nv_scalar(2.0, dtype = "f64")
        ),
        b = nv_scalar(3.0, dtype = "f64")
      ),
      flag = TRUE
    )
  )

  x <- list(a = list(u = 0.5, v = 1.5), b = 2.0)
  expect_quickr_matches_pjrt(graph, x = x, flag = TRUE)
})

test_that("graph_to_quickr_function accepts flat static args (no nested inputs)", {
  skip_if_no_quickr_or_pjrt()

  fn <- function(x, flag) {
    if (flag) {
      x + 1
    } else {
      x - 1
    }
  }

  graph <- trace_fn(
    fn,
    list(
      x = nv_scalar(0.0, dtype = "f64"),
      flag = TRUE
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  f_r <- graph_to_quickr_r_function(graph)
  expect_identical(names(formals(f_quick)), c("x", "flag"))
  expect_identical(names(formals(f_r)), c("x", "flag"))

  out_true <- expect_quickr_matches_pjrt(graph, x = 0.5, flag = TRUE)
  expect_equal(out_true$out_quick, 1.5)
})

test_that("graph_to_quickr_function generates placeholder names for unnamed `...` inputs", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(...) {
      xs <- list(...)
      xs[[1L]] + xs[[2L]]$a + xs[[2L]]$b
    },
    list(
      nv_scalar(0.0, dtype = "f64"),
      list(
        a = nv_scalar(1.0, dtype = "f64"),
        b = nv_scalar(2.0, dtype = "f64")
      )
    )
  )

  # Formals are generated from empty in_tree names; call by position to keep this robust.
  out <- expect_quickr_matches_pjrt(graph, 0.25, list(a = 0.5, b = 1.0))
  expect_equal(out$out_quick, 0.25 + 0.5 + 1.0)
})

test_that("graph_to_quickr_function errors on mismatched flattened input structure", {
  skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x$a$u + x$b
      } else {
        x$a$v + x$b
      }
    },
    list(
      x = list(
        a = list(
          u = nv_scalar(1.0, dtype = "f64"),
          v = nv_scalar(2.0, dtype = "f64")
        ),
        b = nv_scalar(3.0, dtype = "f64")
      ),
      flag = TRUE
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  f_r <- graph_to_quickr_r_function(graph)
  x_bad <- list(a = list(u = 0.5), b = 2.0)
  expect_error(
    f_quick(x = x_bad, flag = TRUE),
    "Expected 4 flattened inputs, got 3",
    fixed = FALSE
  )
  expect_error(
    f_r(x = x_bad, flag = TRUE),
    "Expected 4 flattened inputs, got 3",
    fixed = FALSE
  )
})

test_that("graph_to_quickr_function preserves 1D output dims", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      x + 1
    },
    list(x = nv_tensor(array(0, dim = 3L), dtype = "f64", shape = 3L))
  )

  out <- expect_quickr_matches_pjrt(graph, c(0.5, 1.5, 2.5))
  expect_identical(dim(out$out_quick), c(3L))
})

test_that("graph_to_quickr_function decodes pred leaves when packing outputs", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      list(p = x > 0, v = x + 1)
    },
    list(x = nv_scalar(0.0, dtype = "f64"))
  )

  out <- expect_quickr_matches_pjrt(graph, -0.5)
  out <- out$out_quick

  expect_identical(out$p, FALSE)
  expect_equal(out$v, 0.5)
})
