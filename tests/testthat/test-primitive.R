test_that("prim", {
  expect_equal(prim("add"), p_add)
  expect_true(is_higher_order_primitive(prim("while")))
  p <- Primitive("abc")
  expect_class(p, "anvil::Primitive")
  expect_equal(p@name, "abc")
  expect_snapshot(p)

  on.exit(register_primitive("add", p_add, overwrite = TRUE))
  expect_error(register_primitive("add", p))
  register_primitive("add", p, overwrite = TRUE)
  expect_equal(prim("add"), p)
  expect_list(prim(), types = "anvil::Primitive")
})

test_that("subgraphs extracts subgraphs from higher-order primitives", {
  # Test with p_if
  true_graph <- trace_fn(function() nv_scalar(1), list())
  false_graph <- trace_fn(function() nv_scalar(2), list())
  call <- PrimitiveCall(
    primitive = p_if,
    inputs = list(GraphValue(aval = nv_aten("pred", integer()))),
    params = list(true_graph = true_graph, false_graph = false_graph),
    outputs = list(GraphValue(aval = nv_aten("f32", integer())))
  )

  subgraphs_list <- subgraphs(call)
  expect_length(subgraphs_list, 2L)
  expect_named(subgraphs_list, c("true_graph", "false_graph"))
  expect_identical(subgraphs_list[["true_graph"]], true_graph)
  expect_identical(subgraphs_list[["false_graph"]], false_graph)
})

test_that("subgraphs returns empty list for non-higher-order primitives", {
  call <- PrimitiveCall(
    primitive = p_add,
    inputs = list(
      GraphValue(aval = nv_aten("f32", integer())),
      GraphValue(aval = nv_aten("f32", integer()))
    ),
    params = list(),
    outputs = list(GraphValue(aval = nv_aten("f32", integer())))
  )

  subgraphs_list <- subgraphs(call)
  expect_length(subgraphs_list, 0L)
})

test_that("subgraphs works with p_while", {
  cond_graph <- trace_fn(function(i) i < nv_scalar(10), list(i = nv_scalar(0)))
  body_graph <- trace_fn(function(i) list(i = i + nv_scalar(1)), list(i = nv_scalar(0)))
  call <- PrimitiveCall(
    primitive = p_while,
    inputs = list(GraphValue(aval = nv_aten("f32", integer()))),
    params = list(cond_graph = cond_graph, body_graph = body_graph),
    outputs = list(GraphValue(aval = nv_aten("f32", integer())))
  )

  subgraphs_list <- subgraphs(call)
  expect_length(subgraphs_list, 2L)
  expect_named(subgraphs_list, c("cond_graph", "body_graph"))
  expect_identical(subgraphs_list[["cond_graph"]], cond_graph)
  expect_identical(subgraphs_list[["body_graph"]], body_graph)
})
