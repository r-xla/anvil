test_that("(un)flatten lists", {
  # unnested
  x <- list(a = 1, 2)
  out <- flatten(x)
  expect_equal(out, list(1, 2))
  expect_equal(
    do.call(unflatten, list(build_tree(x), out)),
    x
  )

  # nested depth 1
  x1 <- list(list(1), list(a = 2), 3)
  out1 <- flatten(x1)
  expect_equal(out1, list(1, 2, 3))
  expect_equal(
    do.call(unflatten, list(build_tree(x1), out1)),
    x1
  )

  # nested depth 0
  x2 <- 1L
  out2 <- flatten(x2)
  expect_equal(out2, list(1L))
  expect_equal(
    do.call(unflatten, list(build_tree(x2), out2)),
    x2
  )
})

test_that("flatten_fun", {
  f <- function(a, b) {
    list(a, b)
  }
  args <- list(
    list(list(a = 1), list(2)),
    list(b = -1)
  )

  f_flat <- rlang::exec(flatten_fun, f, !!!args)
  expect_class(f_flat, "anvil::FlattenedFunction")
  out <- rlang::exec(f_flat, !!!flatten(args))
  args_flat <- flatten(args)
  out <- do.call(unflatten, out)
  expect_equal(args, out)
})
