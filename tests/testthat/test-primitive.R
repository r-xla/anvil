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

test_that("new_primitive builds a callable that self-registers into prim_dict", {
  on.exit(rm("np_test", envir = prim_dict))

  fn <- new_primitive("np_test", function(x) x + 1)

  expect_class(fn, "JitPrimitive")
  expect_class(fn, "JitFunction")
  expect_identical(prim("np_test"), fn)
  expect_identical(attr(fn, "primitive")$name, "np_test")
  expect_identical(formals(fn), formals(function(x) x + 1))
})

test_that("new_primitive respects register = FALSE", {
  fn <- new_primitive("np_unregistered", function(x) x, register = FALSE)
  expect_false(exists("np_unregistered", envir = prim_dict, inherits = FALSE))
})

test_that("JitPrimitive [[ delegates to attached AnvilPrimitive", {
  p <- AnvilPrimitive("jp_test_a")
  f <- function(x) x
  attr(f, "primitive") <- p
  class(f) <- c("JitPrimitive", "function")

  f[["stablehlo"]] <- function(x) "stablehlo-rule"
  expect_identical(p[["stablehlo"]](), "stablehlo-rule")

  expect_identical(f[["stablehlo"]], p[["stablehlo"]])
})

test_that("print.JitPrimitive delegates to the AnvilPrimitive", {
  p <- AnvilPrimitive("jp_test_b")
  f <- function(x) x
  attr(f, "primitive") <- p
  class(f) <- c("JitPrimitive", "function")
  expect_output(print(f), "<AnvilPrimitive:jp_test_b>")
})

describe("subgraphs", {
  it("extracts subgraphs from higher-order primitives", {
    true_graph <- trace_fn(function() nv_scalar(1), list())
    false_graph <- trace_fn(function() nv_scalar(2), list())
    call <- PrimitiveCall(
      primitive = p_if,
      inputs = list(GraphValue(aval = nv_aval("bool", integer()))),
      params = list(true_graph = true_graph, false_graph = false_graph),
      outputs = list(GraphValue(aval = nv_aval("f32", integer())))
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
      inputs = list(GraphValue(aval = nv_aval("f32", integer())), GraphValue(aval = nv_aval("f32", integer()))),
      params = list(),
      outputs = list(GraphValue(aval = nv_aval("f32", integer())))
    )
    expect_length(subgraphs(call), 0L)
  })
})
