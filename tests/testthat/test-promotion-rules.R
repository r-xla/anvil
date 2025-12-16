test_that("common_type_info: single argument", {
  s1 <- AbstractTensor(as_dtype("i32"), Shape(c(1, 2)), FALSE)
  result <- common_type_info(s1)
  expect_equal(result[[1L]], as_dtype("i32"))
  expect_equal(result[[2L]], FALSE)

  s2 <- AbstractTensor(as_dtype("f32"), Shape(c(2, 3)), TRUE)
  result <- common_type_info(s2)
  expect_equal(result[[1L]], as_dtype("f32"))
  expect_equal(result[[2L]], TRUE)
})

test_that("common_type_info: two arguments", {
  check <- function(dt1, dt2, a1, a2, expected_dt, expected_ambiguous) {
    s1 <- AbstractTensor(dt1, Shape(c(1, 2)), a1)
    s2 <- AbstractTensor(dt2, Shape(c(2, 1)), a2)
    result <- common_type_info(s1, s2)
    expect_equal(result[[1L]], expected_dt)
    expect_equal(result[[2L]], expected_ambiguous)
    # Check symmetry for dtype (ambiguity may differ based on order in some edge cases)
    result_rev <- common_type_info(s2, s1)
    expect_equal(result_rev[[1L]], expected_dt)
  }

  # both are ambiguous -> result is ambiguous

  check(as_dtype("i32"), as_dtype("i32"), TRUE, TRUE, as_dtype("i32"), TRUE)
  check(as_dtype("i32"), as_dtype("f32"), TRUE, TRUE, as_dtype("f32"), TRUE)

  # one is ambiguous
  # ambiguous float + known int -> ambiguous float (ambiguous wins because it's float)
  check(as_dtype("f32"), as_dtype("i32"), TRUE, FALSE, as_dtype("f32"), TRUE)
  # ambiguous int + known float -> known float (known wins)
  check(as_dtype("i32"), as_dtype("f32"), TRUE, FALSE, as_dtype("f32"), FALSE)
  # both types same -> known wins

  check(as_dtype("i32"), as_dtype("i32"), TRUE, FALSE, as_dtype("i32"), FALSE)

  # neither is ambiguous -> result is not ambiguous
  check(as_dtype("f32"), as_dtype("i32"), FALSE, FALSE, as_dtype("f32"), FALSE)
  check(as_dtype("f32"), as_dtype("f64"), FALSE, FALSE, as_dtype("f64"), FALSE)
  check(as_dtype("ui32"), as_dtype("i32"), FALSE, FALSE, as_dtype("i64"), FALSE)
})

test_that("common_type_info: multiple arguments", {
  i32 <- AbstractTensor(as_dtype("i32"), Shape(1), FALSE)
  f32 <- AbstractTensor(as_dtype("f32"), Shape(2), FALSE)
  f64 <- AbstractTensor(as_dtype("f64"), Shape(3), FALSE)

  result <- common_type_info(i32, f32, f64)
  expect_equal(result[[1L]], as_dtype("f64"))
  expect_equal(result[[2L]], FALSE)

  result <- common_type_info(f64, f32, i32)
  expect_equal(result[[1L]], as_dtype("f64"))
  expect_equal(result[[2L]], FALSE)

  result <- common_type_info(i32, i32, i32)
  expect_equal(result[[1L]], as_dtype("i32"))
  expect_equal(result[[2L]], FALSE)

  # With ambiguous types
  i32_amb <- AbstractTensor(as_dtype("i32"), Shape(1), TRUE)
  i64_known <- AbstractTensor(as_dtype("i64"), Shape(2), FALSE)

  result <- common_type_info(i32_amb, i64_known)
  expect_equal(result[[1L]], as_dtype("i64"))
  expect_equal(result[[2L]], FALSE)
})

test_that("common_type_info: error on no arguments", {
  expect_error(common_type_info(), "No arguments provided")
})

test_that("promote_dt_known", {
  check <- function(dt1, dt2, dt3) {
    expect_equal(
      promote_dt_known(as_dtype(dt1), as_dtype(dt2)),
      as_dtype(dt3)
    )
    expect_equal(
      promote_dt_known(as_dtype(dt2), as_dtype(dt1)),
      as_dtype(dt3)
    )
  }

  check("f64", "f64", "f64")
  check("i32", "i32", "i32")
  check("i1", "i1", "i1")

  # floats dominate
  check("f64", "f32", "f64")
  check("f64", "i32", "f64")
  check("f32", "i32", "f32")
  check("f32", "i1", "f32")

  # signed ints
  check("i32", "i16", "i32")
  check("i64", "i32", "i64")
  check("i64", "i16", "i64")
  check("i64", "i1", "i64")
  # against unsigned ints
  check("i32", "ui8", "i32")
  check("i32", "ui32", "i64")
  check("i64", "ui64", "i64")
  # unsigned vs unsigned
  check("ui64", "ui32", "ui64")
})

test_that("promote_dt_ambiguous", {
  check <- function(x, y, z) {
    expect_equal(
      promote_dt_ambiguous(as_dtype(x), as_dtype(y)),
      as_dtype(z)
    )
    expect_equal(
      promote_dt_ambiguous(as_dtype(y), as_dtype(x)),
      as_dtype(z)
    )
  }
  check("i32", "i32", "i32")
  check("f32", "f32", "f32")
  check("i1", "i1", "i1")

  check("i32", "f32", "f32")
  check("i1", "f32", "f32")
  check("i1", "i32", "i32")
})

test_that("promote_dt_ambiguous_to_known", {
  check <- function(amb, known, z) {
    expect_equal(
      promote_dt_ambiguous_to_known(as_dtype(amb), as_dtype(known)),
      as_dtype(z)
    )
  }
  check("i32", "i32", "i32")
  check("i1", "i1", "i1")
  check("f32", "f32", "f32")
  # ambiguous can only be i32, f32 or bool

  check("i32", "i8", "i8")
  check("i32", "i64", "i64")
  check("i32", "i1", "i32")
  check("i64", "f32", "f32")

  check("f32", "f64", "f64")
  check("f64", "f32", "f32")
  check("f32", "i32", "f32")

  check("i1", "f32", "f32")
  check("i1", "i32", "i32")
})
