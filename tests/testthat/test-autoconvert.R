test_that("jit: autoconverts length-1 numeric scalar to ambiguous nv_scalar", {
  f <- jit(identity)
  out <- f(1)
  expect_equal(out, nv_scalar(1, ambiguous = TRUE))
})

test_that("jit: ambiguous autoconverted scalar + literal stays ambiguous", {
  f <- jit(\(x) x + 1)
  out <- f(1)
  expect_equal(dtype(out), as_dtype("f32"))
  expect_equal(shape(out), integer())
  expect_true(ambiguous(out))
})

test_that("jit: autoconverts length-1 integer scalar to ambiguous nv_scalar", {
  f <- jit(identity)
  out <- f(1L)
  expect_equal(out, nv_scalar(1L, ambiguous = TRUE))
})

test_that("jit: autoconverts length-1 logical scalar to ambiguous nv_scalar", {
  f <- jit(identity)
  out <- f(TRUE)
  expect_equal(out, nv_scalar(TRUE, ambiguous = TRUE))
})

test_that("jit: promotes scalar dtype against non-ambiguous typed scalar", {
  f <- jit(\(x, y) x + y)
  out <- f(1, nv_scalar(2, dtype = "f64"))
  expect_equal(dtype(out), as_dtype("f64"))
  expect_false(ambiguous(out))
})

test_that("jit: autoconverts matrix via nv_array (ambiguous)", {
  f <- jit(identity)
  out <- f(matrix(1:4, 2, 2))
  expect_equal(out, nv_array(matrix(1:4, 2, 2), ambiguous = TRUE))
})

test_that("jit: autoconverts higher-dim array via nv_array", {
  f <- jit(identity)
  a <- array(1:24, dim = c(2, 3, 4))
  out <- f(a)
  expect_equal(dtype(out), as_dtype("i32"))
  expect_equal(shape(out), c(2L, 3L, 4L))
})

test_that("jit: bare vector without dim errors", {
  f <- jit(function(x) x)
  expect_snapshot(f(c(1, 2, 3)), error = TRUE)
})

test_that("jit: non-array/non-scalar leaves (e.g. character) error", {
  f <- jit(function(x) x)
  expect_snapshot(f("hello"), error = TRUE)
})

test_that("jit: nested list is flattened; leaves are autoconverted", {
  f <- jit(function(pair) pair[[1]] + pair[[2]])
  out <- f(list(1, 2))
  expect_equal(dtype(out), as_dtype("f32"))
  expect_equal(shape(out), integer())
  expect_equal(as_array(out), 3)
})

test_that("jit: static args are not autoconverted", {
  f <- jit(function(x, flag) if (flag) x + 1 else x * 2, static = "flag")
  out <- f(nv_scalar(3), TRUE)
  expect_equal(as_array(out), 4)
  out2 <- f(3, FALSE)
  expect_equal(as_array(out2), 6)
})

test_that("jit: inside trace, autoconvert does not fire on GraphValues", {
  inner <- jit(function(x) x + 1)
  outer <- jit(function(x) inner(x))
  out <- outer(nv_scalar(1))
  expect_equal(as_array(out), 2)
})

test_that("xla: autoconverts scalar and matrix inputs", {
  f_compiled <- xla(
    function(x, y) x + y,
    args = list(x = nv_abstract("f32", c()), y = nv_abstract("f32", c(2, 2)))
  )
  out <- f_compiled(1, matrix(c(1, 2, 3, 4), 2, 2))
  expect_equal(dtype(out), as_dtype("f32"))
  expect_equal(shape(out), c(2L, 2L))
})

test_that("xla: bare vector errors", {
  f_compiled <- xla(
    function(x) x,
    args = list(x = nv_abstract("f32", c(3)))
  )
  expect_snapshot(f_compiled(c(1, 2, 3)), error = TRUE)
})

test_that("xla: accepts tree (nested list) inputs", {
  f_compiled <- xla(
    function(pair) pair[[1]] + pair[[2]],
    args = list(pair = list(nv_abstract("f32", c()), nv_abstract("f32", c())))
  )
  out <- f_compiled(list(1, nv_scalar(2, dtype = "f32")))
  expect_equal(dtype(out), as_dtype("f32"))
  expect_equal(shape(out), integer())
  expect_equal(as_array(out), 3)
})

test_that("jit_eval: scalar expression works unchanged", {
  expect_equal(as_array(jit_eval(nv_scalar(1) + nv_scalar(2))), 3)
})

test_that("quickr: autoconverts scalar input", {
  skip_if_not_installed("quickr")
  local_backend("quickr")
  f <- jit(identity)
  out <- f(1)
  expect_equal(out, nv_scalar(1, ambiguous = TRUE))
})

test_that("quickr: autoconverts matrix input", {
  skip_if_not_installed("quickr")
  local_backend("quickr")
  f <- jit(identity)
  out <- f(matrix(1:4, 2, 2))
  expect_equal(dtype(out), as_dtype("i32"))
  expect_equal(shape(out), c(2L, 2L))
  expect_equal(out, nv_array(matrix(1:4, 2, 2), ambiguous = TRUE))
})

test_that("quickr: nested input tree with mixed AnvilArray/scalar still works", {
  skip_if_not_installed("quickr")
  local_backend("quickr")
  f <- jit(function(pair) pair[[1]] + pair[[2]])
  out <- f(list(nv_scalar(1L), 2L))
  expect_equal(out, nv_scalar(3L))
})

test_that("quickr: bare vector errors", {
  skip_if_not_installed("quickr")
  local_backend("quickr")
  f <- jit(function(x) x)
  expect_snapshot(f(c(1, 2, 3)), error = TRUE)
})

test_that("tree_paths: scalar argument", {
  tree <- build_tree(list(x = 1))
  expect_equal(tree_paths(tree), "x")
})

test_that("tree_paths: named nested list", {
  tree <- build_tree(list(l = list(a = 1, b = 2)))
  expect_equal(tree_paths(tree), c("l$a", "l$b"))
})

test_that("tree_paths: unnamed nested list", {
  tree <- build_tree(list(l = list(1, 2)))
  expect_equal(tree_paths(tree), c("l[[1]]", "l[[2]]"))
})

test_that("tree_paths: mixed named and unnamed", {
  tree <- build_tree(list(l = list(1, b = 2)))
  expect_equal(tree_paths(tree), c("l[[1]]", "l$b"))
})

test_that("tree_paths: deeply nested", {
  tree <- build_tree(list(l = list(list(a = 1))))
  expect_equal(tree_paths(tree), "l[[1]]$a")
})

test_that("tree_paths: multiple top-level args", {
  tree <- build_tree(list(x = 1, y = 2))
  expect_equal(tree_paths(tree), c("x", "y"))
})

test_that("tree_paths: top-level list arg with nested list", {
  tree <- build_tree(list(pair = list(list(a = 1), list(b = 2))))
  expect_equal(tree_paths(tree), c("pair[[1]]$a", "pair[[2]]$b"))
})

test_that("jit: error shows path for nested list element", {
  f <- jit(function(l) l[[1]])
  expect_snapshot(f(list(list(a = "abc"))), error = TRUE)
})

test_that("jit: error shows path for unnamed nested element", {
  f <- jit(function(pair) pair[[1]])
  expect_snapshot(f(list("bad", nv_scalar(1))), error = TRUE)
})

test_that("xla: error shows path for nested list element", {
  f_compiled <- xla(
    function(pair) pair[[1]] + pair[[2]],
    args = list(pair = list(nv_abstract("f32", c()), nv_abstract("f32", c())))
  )
  expect_snapshot(f_compiled(list("bad", nv_scalar(1))), error = TRUE)
})
