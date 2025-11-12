test_that("basic test", {
  f <- function(x, y) {
    x + y
  }

  out <- graphify(f, list(x = nv_scalar(1), y = nv_scalar(2)))
  graph <- out[[1]]
  inputs <- out[[2]]
  expect_equal(inputs[[1]], ShapedTensor(dt_f32, Shape(c())))
  expect_equal(inputs[[2]], ShapedTensor(dt_f32, Shape(c())))
  expect_true(is_graph(graph))

  graph_fn <- graph_to_function(graph)
  expect_equal(graph_fn(inputs[[1]], inputs[[2]]), 3)
})

test_that("sub-graph", {
  # TODO: Need gradient() for this
})

test_that("nested inputs and outputs", {
  f <- function(lst) {
    list(lst[[1]] + lst[[2]])
  }

  out <- graphify(f, list(lst = list(nv_scalar(1), nv_scalar(2))))
  graph <- out[[1]]
  inputs <- out[[2]]
  expect_equal(inputs[[1]], ShapedTensor(dt_f32, Shape(c())))
  expect_equal(inputs[[2]], ShapedTensor(dt_f32, Shape(c())))
  expect_true(is_graph(graph))
})

test_that("graph_to_function", {
})

test_that("closed-over constants", {
  f <- function(c) {

  }
})


test_that("captured constants are added to inputs of graph", {
})

test_that("can call graph function", {
})

test_that("graph can call into other graph", {
})

test_that("local_graph creates a graph", {
  globals[["CURRENT_GRAPH"]] <- NULL
  g <- local_graph()
  expect_false(is.null(globals[["CURRENT_GRAPH"]]))
  expect_true(inherits(g, "anvil::Graph"))
})

test_that("local_graph restores previous graph", {
  globals[["CURRENT_GRAPH"]] <- NULL
  g1 <- local_graph()
  inner_test <- function() {
    g2 <- local_graph()
    (function() local_graph())()
    expect_equal(.current_graph(), g2)
  }
  inner_test()
  expect_equal(g1, .current_graph())
})

test_that(".current_graph errors when no graph exists", {
  globals[["CURRENT_GRAPH"]] <- NULL
  expect_error(.current_graph(), "No graph is currently being built")
})
