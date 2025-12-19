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
