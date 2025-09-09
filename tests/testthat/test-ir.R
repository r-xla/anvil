test_that("can construct Variable, Literal, Atom, Equation, Expr", {
  dt <- stablehlo::FloatType("f32")
  shp <- stablehlo::Shape(1L)

  sa <- ShapedArray(dtype = dt, shape = shp)
  v1 <- Variable(sa)

  lit_arr <- nvl_array(1.0)
  l1 <- Literal(lit_arr)

  expect_true(inherits(v1, "Variable"))
  expect_true(inherits(l1, "Literal"))
  expect_true(inherits(v1, Atom))
  expect_true(inherits(l1, Atom))

  vout <- Variable(sa)
  eq <- Equation(
    primitive = op_add,
    inputs = Atoms(list(v1, l1)),
    params = Params(list()),
    out_binders = Variables(list(vout))
  )
  expect_true(inherits(eq, "Equation"))

  expr <- Expr(
    in_binders = Variables(list(v1)),
    equations = Equations(list()),
    outputs = Variables(list(v1))
  )
  expect_true(inherits(expr, "Expr"))
})

test_that("typecheck_expr works for expr with no equations and bound outputs", {
  dt <- stablehlo::FloatType("f32")
  shp <- stablehlo::Shape(1L)
  sa <- ShapedArray(dtype = dt, shape = shp)
  v1 <- Variable(sa)

  expr <- Expr(
    in_binders = Variables(list(v1)),
    equations = Equations(list()),
    outputs = Variables(list(v1))
  )

  et <- typecheck_expr(expr)
  expect_true(inherits(et, "ExprType"))
})

test_that("expr_to_function returns a function that passes through inputs when no equations", {
  dt <- stablehlo::FloatType("f32")
  shp <- stablehlo::Shape(1L)
  sa <- ShapedArray(dtype = dt, shape = shp)
  v1 <- Variable(sa)
  v2 <- Variable(sa)

  expr <- Expr(
    in_binders = Variables(list(v1, v2)),
    equations = Equations(list()),
    outputs = Variables(list(v1, v2))
  )

  f <- expr_to_function(expr)
  a <- nvl_array(1.0)
  b <- nvl_array(2.0)
  out <- f(a, b)
  expect_length(out, 2L)
  expect_true(identical(out[[1]], a))
  expect_true(identical(out[[2]], b))
})
