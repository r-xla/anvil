test_that("literals", {
  f <- function(x) {
    x * 1L
  }
  graph <- trace_fn(f, list(x = nv_scalar(1)))
  expect_snapshot(graph)

  # higher-dimensional literals
  f <- function() {
    nv_fill(1, shape = c(2, 1))
  }
  graph <- trace_fn(f, list())
  expect_snapshot(graph)
})

test_that("ambiguity is printed via ?", {
  f <- function(x) {
    x * 1
  }
  graph <- trace_fn(f, list(x = nv_scalar(TRUE)))
  expect_snapshot(graph)
})

test_that("Graph: printing", {
  f <- function(x, y) {
    nvl_add(x, y)
  }
  graph <- trace_fn(f, list(x = nv_scalar(1), y = nv_scalar(2)))
  expect_snapshot(graph)
  # with param
  f1 <- function(x) {
    mean(x)
  }
  graph1 <- trace_fn(f1, list(x = nv_tensor(1:10, dtype = "f32", shape = c(2, 5))))
  expect_snapshot(graph1)
})

test_that("constants", {
  y <- nv_scalar(1)
  f <- function(x) {
    x + y
  }
  graph <- trace_fn(f, list(x = nv_scalar(2)))
  expect_snapshot(graph)
})

test_that("sub-graphs (if)", {
  f <- function(x) {
    nv_if(x, nv_scalar(1), nv_scalar(2))
  }
  graph <- trace_fn(f, list(x = nv_scalar(TRUE)))
  expect_snapshot(graph)
})

test_that("sub-graphs (while)", {
  f <- function(x) {
    nv_while(list(i = nv_scalar(0)), \(i) i < x, \(i) {
      list(i = i + nv_scalar(1))
    })
  }
  graph <- trace_fn(f, list(x = nv_scalar(10)))
  expect_snapshot(graph)
})

test_that("params", {
  f <- function(x) {
    nv_reduce_max(x, dims = 1, drop = TRUE)
  }
  graph <- trace_fn(f, list(x = nv_tensor(1:10)))
  expect_snapshot(graph)
})
