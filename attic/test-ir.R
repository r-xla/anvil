test_that("IRVariable", {
  var <- IRVariable(
    aval = ShapedTensor(FloatType("f32"), Shape(1L))
  )
  expect_class(var, "anvil::IRVariable")
  expect_snapshot(var)
})

test_that("IRVariables", {
  vars <- IRVariables(list(
    IRVariable(ShapedTensor(FloatType("f32"), Shape(1L))),
    IRVariable(ShapedTensor(FloatType("f32"), Shape(1L)))
  ))
  expect_class(vars, "anvil::IRVariables")
  expect_snapshot(vars)
})

test_that("IRLiteral", {
  expect_error(IRLiteral(nv_tensor(1L)), "scalars")
  lit <- IRLiteral(
    value = nv_scalar(1L)
  )
  expect_class(lit, "anvil::IRLiteral")
  expect_snapshot(lit)
})

test_that("IRAtoms", {
  atoms <- IRAtoms(list(
    IRVariable(ShapedTensor(FloatType("f32"), Shape(1L))),
    IRLiteral(nv_scalar(1L))
  ))
  expect_class(atoms, "anvil::IRAtoms")
  expect_snapshot(atoms)
})

test_that("IREquation", {
  eqn <- IREquation(
    prim = nvl_add,
    inputs = IRAtoms(list(
      IRVariable(ShapedTensor(FloatType("f32"), Shape(1L))),
      IRLiteral(nv_scalar(1L))
    )),
    params = IRParams(list(1L)),
    out_binders = IRVariables(list(IRVariable(ShapedTensor(
      FloatType("f32"),
      Shape(1L)
    ))))
  )
  expect_class(eqn, "anvil::IREquation")
  expect_snapshot(eqn)
})

test_that("IREquations & IRExpr", {
  v <- IRVariable(ShapedTensor(FloatType("f32"), Shape(integer())))
  l <- IRLiteral(nv_scalar(1L))
  z <- IRVariable(ShapedTensor(FloatType("f32"), Shape(integer())))
  y <- IRVariable(ShapedTensor(FloatType("f32"), Shape(integer())))

  eqns <- IREquations(list(
    IREquation(
      primitive = nvl_add,
      inputs = IRAtoms(list(v, l)),
      out_binders = IRVariables(list(z)),
    ),
    IREquation(
      primitive = nvl_add,
      inputs = IRAtoms(list(
        IRVariable(ShapedTensor(FloatType("f32"), Shape(1L))),
        IRLiteral(nv_scalar(1L))
      )),
      params = IRParams(list()),
      out_binders = IRVariables(list(y))
    )
  ))
  expect_class(eqns, "anvil::IREquations")
  expect_snapshot(cat(repr(eqns)))

  expr <- IRExpr(
    in_binders = IRVariables(list(v)),
    equations = eqns,
    outputs = IRVariables(list(y))
  )
  expect_class(expr, "anvil::IRExpr")
  expect_snapshot(expr)
})
