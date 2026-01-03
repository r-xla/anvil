test_that("trace_fn: simple test", {
  f <- function(x, y) {
    nvl_add(x, y)
  }
  graph <- trace_fn(f, list(x = nv_scalar(1), y = nv_scalar(2)))
  expect_true(is_graph(graph))
  expect_list(graph@inputs, len = 2L, types = "anvil::mut<GraphValue>")
  expect_list(graph@calls, len = 1L, types = "anvil::PrimitiveCall")
  expect_list(graph@outputs, len = 1L, types = "anvil::mut<GraphValue>")
  expect_true(identical(graph@outputs, graph@calls[[1]]@outputs))
})

test_that("trace_fn: in- and outputs are reference identical to the outputs of the calls that produced them", {
  f <- function(x, y) {
    nvl_add(x, y)
  }
  graph <- trace_fn(f, list(x = nv_scalar(1), y = nv_scalar(2)))
  expect_true(identical(graph@outputs, graph@calls[[1]]@outputs))
  expect_true(identical(graph@inputs, graph@calls[[1]]@inputs))
})

test_that("trace_fn: nested inputs and outputs", {
  f <- function(lst) {
    list(nvl_add(lst[[1]], lst[[2]]))
  }

  graph <- trace_fn(f, list(lst = list(nv_scalar(1), nv_scalar(2))))
  expect_list(graph@inputs, len = 2L, types = "anvil::mut<GraphValue>")
  expect_list(graph@calls, len = 1L, types = "anvil::PrimitiveCall")
  expect_list(graph@outputs, len = 1L, types = "anvil::mut<GraphValue>")
  expect_equal(
    unflatten(graph@in_tree, list(1, 2)),
    list(lst = list(1, 2))
  )
  expect_equal(
    unflatten(graph@out_tree, 1),
    list(1)
  )
})

test_that("trace_fn: closed-over constants", {
  x <- nv_scalar(1)
  f <- function(y) {
    nvl_add(x, y)
  }
  graph <- trace_fn(f, list(y = nv_scalar(2)))
  expect_list(graph@inputs, len = 1L, types = "anvil::mut<GraphValue>")
  expect_list(graph@calls, len = 1L, types = "anvil::PrimitiveCall")
  expect_list(graph@outputs, len = 1L, types = "anvil::mut<GraphValue>")

  # What do we expect here?
  # We want the resulting graph to have a constant and two inputs

  expect_true(is_graph_value(graph@calls[[1]]@inputs[[1]]))
  expect_true(is_graph_value(graph@calls[[1]]@inputs[[2]]))
  expect_true(identical(x, graph@constants[[1]]@aval@data))
  expect_equal(length(graph@constants), 1L)
})

test_that("trace_fn can deduplicate constants", {
  x <- nv_scalar(1)
  f <- function(y) {
    nvl_add(x, x)
  }
  graph <- trace_fn(f, list(y = nv_scalar(2)))
  expect_equal(length(graph@constants), 1L)
  expect_identical(graph@constants[[1]]@aval@data, x)
})

test_that("trace_fn works without arguments", {
  # For this it is necessary to also box outputs in trace_fn()
  x <- nv_scalar(1)
  f <- function() {
    x
  }
  graph <- trace_fn(f, list())
  expect_equal(length(graph@inputs), 0L)
  expect_equal(length(graph@outputs), 1L)
  expect_identical(graph@outputs[[1]]@aval@data, x)
  expect_equal(length(graph@outputs), 1L)
  expect_equal(length(graph@calls), 0L)
})


test_that("local_descriptor creates a graph", {
  globals[["CURRENT_DESCRIPTOR"]] <- NULL
  g <- local_descriptor()
  expect_false(is.null(globals[["CURRENT_DESCRIPTOR"]]))
  expect_true(is_graph_descriptor(globals[["CURRENT_DESCRIPTOR"]]))
})

test_that("local_descriptor restores previous graph", {
  globals[["CURRENT_DESCRIPTOR"]] <- NULL
  g1 <- local_descriptor()
  inner_test <- function() {
    g2 <- local_descriptor()
    (function() local_descriptor())()
    expect_equal(.current_descriptor(), g2)
  }
  inner_test()
  expect_equal(g1, .current_descriptor())
})

test_that(".current_descriptor errors when no graph exists", {
  globals[["CURRENT_DESCRIPTOR"]] <- NULL
  expect_error(.current_descriptor(), "No graph is currently being built")
})

test_that("constants: same tensor is constant and input at the same time", {
  # Not sure what we want to happen here.
  f <- jit(function(x) {
    h <- function(x) x * y
    y <- nv_scalar(2)
    gradient(h)(y)
  })
  f(nv_scalar(1))
})

test_that("closed-over constant is passed as argument to transformation", {
  x <- nv_scalar(1)
  f <- jit(function() {
    h <- function(y) y * y
    gradient(h)(x)
  })
  f()
})

test_that("can pass constant to nested trace_fn call if it does not exist in the parent graph", {
  f <- jit(function() {
    g <- function(y) y * y
    gradient(g)(nv_scalar(2))
  })
  expect_equal(f(), list(y = nv_scalar(4)))
})

test_that("can pass constant to nested trace_fn call if it is defined in the parent graph", {
  f <- jit(function() {
    nv_add(y, y)
    g <- function(y) y * y
    gradient(g)(y)
  })
  y <- nv_scalar(2)
  expect_equal(f(), list(y = nv_scalar(4)))
})

test_that("GraphLiteral", {
  gl <- GraphLiteral(LiteralTensor(1L, integer(), ambiguous = TRUE))
  expect_equal(dtype(gl), as_dtype("i32"))
  expect_equal(shape(gl), integer())
  expect_snapshot(gl)
})

test_that("trace_fn works with nv_aten inputs", {
  f <- function(x, y) {
    nvl_add(x, y)
  }
  in_type <- nv_aten("f32", c(2, 2))
  graph <- trace_fn(f, list(x = in_type, y = in_type))
  expect_true(is_graph(graph))
  expect_equal(graph@inputs[[1L]]@aval, in_type)
  expect_equal(graph@inputs[[2L]]@aval, in_type)
  expect_equal(length(graph@inputs), 2L)
  expect_equal(length(graph@calls), 1L)
  expect_equal(length(graph@outputs), 1L)
  expect_equal(graph@calls[[1L]]@primitive, p_add)
})

test_that("local_descriptor errors when run in the global environment", {
  expect_error(eval(quote(local_descriptor()), globalenv()), "Don't run local_descriptor in the global environment")
})

test_that("can pass abstract tensors to trace_fn", {
  # Here, its fine because we call into maybe_box_input, which will convert the abstract tensor
  # into a GraphValue/Box before any infix op can be called
  f <- function(x, y) {
    nvl_add(x, y)
  }
  in_type <- nv_aten("f32", c(2, 2))
  graph <- trace_fn(f, list(x = in_type, y = in_type))
  expect_true(is_graph(graph))
  expect_equal(graph@inputs[[1L]]@aval, in_type)
  expect_equal(graph@inputs[[2L]]@aval, in_type)
  expect_equal(length(graph@inputs), 2L)
})
