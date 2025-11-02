test_that("graph interpretation", {
  f <- function(x, y) {
    x + y
  }

  graph <- graphify(f, x = nv_scalar(1), y = nv_scalar(2))

})

test_that("closed-over constants", {
  f <- function(c) {

  }
})
