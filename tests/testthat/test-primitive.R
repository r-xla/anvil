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

test_that("quickr rules are exposed through primitives", {
  expect_true(is.function(prim("add")[["quickr"]]))
  expect_null(prim("print")[["quickr"]])
})

documented_primitive_ids <- function() {
  primitives_path <- testthat::test_path("..", "..", "R", "primitives.R")
  if (!file.exists(primitives_path)) {
    testthat::skip("R/primitives.R is only available when testing from package source")
  }

  primitive_lines <- readLines(primitives_path)
  sub(
    "^#' @templateVar primitive_id ",
    "",
    grep("^#' @templateVar primitive_id ", primitive_lines, value = TRUE)
  )
}

test_that("documented primitive ids resolve to registered primitives", {
  primitive_ids <- documented_primitive_ids()

  missing <- primitive_ids[vapply(primitive_ids, function(id) is.null(prim(id)), logical(1))]
  expect_identical(missing, character())
})

describe("subgraphs", {
  it("extracts subgraphs from higher-order primitives", {
    true_graph <- trace_fn(function() nv_scalar(1), list())
    false_graph <- trace_fn(function() nv_scalar(2), list())
    call <- PrimitiveCall(
      primitive = p_if,
      inputs = list(GraphValue(aval = nv_aten("bool", integer()))),
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
