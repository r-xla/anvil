describe("nv_mean", {
  it("computes gradient", {
    f <- function(y, alpha) {
      mean(y - alpha)
    }

    alpha <- nv_scalar(0.5)
    y <- nv_tensor(1:10, "f32")
    out <- jit(gradient(f, wrt = "alpha"))(y, alpha)
    expect_equal(shape(out[[1L]]), integer())
  })
})

describe("nv_get_elt", {
  it("computes gradient correctly", {
    f <- function(x) {
      idx1 <- nv_scalar(2L, dtype = "i32")
      idx2 <- nv_scalar(2L, dtype = "i32")
      nv_get_elt(x, idx1, idx2)
    }

    g <- jit(gradient(f))
    x <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
    grads <- g(x)

    # Gradient should be 1 at position (2, 2) and 0 elsewhere
    expected <- matrix(0, nrow = 3, ncol = 4)
    expected[2, 2] <- 1
    expect_equal(as_array(grads[[1L]]), expected)
  })
})

describe("nv_set_elt", {
  it("computes gradient correctly", {
    f <- function(x) {
      idx1 <- nv_scalar(2L, dtype = "i32")
      idx2 <- nv_scalar(3L, dtype = "i32")
      value <- nv_scalar(100, dtype = "f32")
      updated <- nv_set_elt(x, idx1, idx2, value = value)
      nv_reduce_sum(updated, dims = c(1L, 2L), drop = TRUE)
    }

    g <- jit(gradient(f))
    x <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
    grads <- g(x)

    # Gradient should be 1 everywhere except position (2, 3) which should be 0
    expected <- matrix(1, nrow = 3, ncol = 4)
    expected[2, 3] <- 0
    expect_equal(as_array(grads[[1L]]), expected)
  })
})
