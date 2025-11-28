test_that("remove_dead_code removes dead code", {
  f <- function(x, y) {
    # This addition doesn't contribute to the output
    dead <- nvl_add(x, y)
    # Only this multiplication is used in the output
    nvl_mul(x, y)
  }
  graph <- graphify(f, list(x = nv_scalar(3), y = nv_scalar(4)))

  # Should only have the multiply call, not the add call
  expect_equal(length(graph@calls), 1L)
  expect_equal(graph@calls[[1L]]@primitive@name, "mul")
})

#test_that("remove_dead_code removes dead constants", {
#  dead_const <- nv_scalar(99)
#  used_const <- nv_scalar(2)
#
#  f <- function(x) {
#    # dead_const is not used in output
#    dead <- nvl_add(x, dead_const)
#    # only used_const contributes
#    nvl_mul(x, used_const)
#  }
#  graph <- graphify(f, list(x = nv_scalar(3)))
#
#  # Should only have the multiply call
#  expect_equal(length(graph@calls), 1L)
#  expect_equal(graph@calls[[1L]]@primitive@name, "mul")
#
#  # Should only have the used constant
#  expect_equal(length(graph@constants), 1L)
#  expect_equal(as.numeric(as_array(graph@constants[[1L]]@aval@data)), 2)
#})

test_that("remove_dead_code keeps transitively needed calls", {
  f <- function(x, y) {
    # This is needed because it's used by the second operation
    intermediate <- nvl_add(x, y)
    nvl_mul(intermediate, x)
  }
  graph <- graphify(f, list(x = nv_scalar(3), y = nv_scalar(4)))

  # Both calls should be kept
  expect_equal(length(graph@calls), 2L)
})
