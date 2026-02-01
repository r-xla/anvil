test_that("prim", {
  expect_equal(prim("add"), p_add)
  expect_true(is_higher_order_primitive(prim("while")))
  p <- AnvilPrimitive("abc")
  expect_class(p, "AnvilPrimitive")
  expect_equal(p$name, "abc")
  expect_snapshot(p)

  on.exit(register_primitive("add", p_add, overwrite = TRUE))
  expect_error(register_primitive("add", p))
  register_primitive("add", p, overwrite = TRUE)
  expect_equal(prim("add"), p)
  expect_list(prim(), types = "AnvilPrimitive")
})

describe("subgraphs", {
  it("extracts subgraphs from higher-order primitives", {
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
  it("returns empty list for non-higher-order primitives", {
    call <- PrimitiveCall(
      primitive = p_add,
      inputs = list(GraphValue(aval = nv_aten("f32", integer())), GraphValue(aval = nv_aten("f32", integer()))),
      params = list(),
      outputs = list(GraphValue(aval = nv_aten("f32", integer())))
    )
    expect_length(subgraphs(call), 0L)
  })
})
