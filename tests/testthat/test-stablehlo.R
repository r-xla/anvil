test_that("stablehlo: basic test", {
  graph <- graphify(nvl_add, list(lhs = nv_tensor(1), rhs = nv_tensor(2)))
  # expect_identical() is buggy ...
  expect_true(identical(graph@outputs, graph@calls[[1]]@outputs))
  out <- stablehlo(graph)
})

test_that("stablehlo: a constant", {
  x <- nv_scalar(1)
  f <- function(y) {
    x + y
  }
  graph <- graphify(f, list(y = nv_scalar(2)))
  out <- stablehlo(graph)
  func <- out[[1L]]
  const <- out[[2L]]
  expect_identical(const, list(x))
})
