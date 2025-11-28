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
  const <- out[[2L]][[1L]]
  expect_true(is_graph_value(const))
  expect_identical(const@aval@data, x)
})

test_that("donate: simple example", {
  graph <- graphify(identity, list(x = nv_tensor(3:4, dtype = "i32")))
  out <- stablehlo(graph, donate = "x")
  expect_equal(out[[1]]@inputs@items[[1]]@alias, 0L)
})

test_that("donate: multiple inputs, only some donated", {
  f <- function(x, y) x + y
  graph <- graphify(f, list(
    x = nv_tensor(array(1:4, c(2, 2))),
    y = nv_tensor(array(5:8, c(2, 2)))
  ))
  out <- stablehlo(graph, donate = "x")
  expect_true(
    out[[1]]@inputs@items[[1]]@alias == 0L || out[[1]]@inputs@items[[2]]@alias == 0L
  )
})

test_that("donate: nested list inputs", {
  f <- function(x) list(x[[1]], x[[2]])
  graph <- graphify(f, list(x = list(nv_tensor(1), nv_tensor(2))))
  out <- stablehlo(graph, donate = "x")
  expect_permutation(
    sapply(1:2, \(i) out[[1]]@inputs@items[[i]]@alias),
    c(0L, 1L)
  )
})
