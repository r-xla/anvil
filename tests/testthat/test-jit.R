test_that("jit: basic test", {
  f <- jit(function(x, y) {
    nvl_add(x, y)
  })

  expect_equal(
    f(nv_scalar(1), nv_scalar(2)),
    nv_scalar(3)
  )
  expect_equal(
    f(nv_scalar(1), nv_scalar(2)),
    nv_scalar(3)
  )
})

test_that("jit: a constant", {
  x <- nv_scalar(1)
  f <- function(y) {
    nvl_add(x, y)
  }
  f_jit <- jit(f)
  expect_equal(
    f_jit(nv_scalar(2)),
    nv_scalar(3)
  )
  x <- nv_scalar(2)
  # the constant is now saved in f_jit, so new x is not found
  cache_size(f_jit)
  expect_equal(
    f_jit(nv_scalar(2)),
    nv_scalar(3)
  )
  cache_size(f_jit)
})

test_that("jit basic test", {
  f <- function(x, y) {
    x + y
  }
  f_jit <- jit(f)

  expect_equal(
    f_jit(nv_array(1), nv_array(2)),
    nv_array(3)
  )
  # cache hit:
  expect_equal(
    f_jit(nv_array(1), nv_array(2)),
    nv_array(3)
  )
})

test_that("can return single array", {
  f <- jit(nvl_add)
  expect_equal(
    f(
      nv_array(1.0),
      nv_array(-2.0)
    ),
    nv_array(-1.0)
  )
})

test_that("can return nested list", {
  f <- jit(\(x, y) {
    list(x, list(a = y))
  })
  x <- nv_array(1)
  y <- nv_array(2)
  expect_equal(
    f(x, y),
    list(x, list(a = y))
  )
})

test_that("can take in nested list", {
  f <- jit(\(x) {
    x$a
  })
  x <- nv_array(1)
  expect_equal(
    f(list(a = x)),
    x
  )
})

test_that("Can multiply values from lists", {
  f <- jit(function(x, y) {
    x[[1]] * y[[1]]
  })
  expect_equal(
    f(list(nv_scalar(2)), list(nv_scalar(3))),
    nv_scalar(6)
  )
})

test_that("multiple returns", {
  f <- jit(function(x, y) {
    list(
      nvl_add(x, y),
      nvl_mul(x, y)
    )
  })

  out <- f(
    nv_array(1.0),
    nv_array(2.0)
  )
  expect_equal(
    out[[1]],
    nv_array(3.0)
  )
  expect_equal(
    out[[2]],
    nv_array(2.0)
  )
})

test_that("jitted function has class JitFunction", {
  f_jit <- jit(function(x) x)
  expect_s3_class(f_jit, "JitFunction")
})

test_that("jit(jit(f)) works (#220)", {
  f_jit <- jit(function(x) x + nv_scalar(1L))
  f_jit2 <- jit(f_jit)
  expect_equal(f_jit2(nv_array(3L)), nv_array(4L))
})

test_that("keeps argument names", {
  f1 <- function(x, y) {
    x + y
  }
  f1_jit <- jit(f1)
  expect_equal(
    formals(f1_jit),
    formals(f1)
  )
  f1_jit(nv_array(1), nv_array(2))

  expect_equal(
    f1_jit(nv_array(1), nv_array(2)),
    nv_array(3)
  )
  f2 <- function(x, y = nv_array(1)) {
    x + y
  }
  f2_jit <- jit(f2)
  expect_equal(
    formals(f2_jit),
    formals(f2)
  )
  expect_equal(
    f2_jit(nv_array(1), nv_array(2)),
    nv_array(3)
  )
  f3 <- function(a, ...) {
    Reduce(`+`, list(...)) * a
  }
  f3_jit <- jit(f3)
  expect_equal(
    formals(f3_jit),
    formals(f3)
  )
  expect_equal(
    f3_jit(nv_array(2), nv_array(3), nv_array(4)),
    nv_array(14)
  )
})

test_that("can mark arguments as static ", {
  f <- jit(
    function(x, add_one) {
      if (add_one) {
        x + nv_array(1)
      } else {
        x
      }
    },
    static = "add_one"
    # TODO: Better error message ...
  )
  expect_equal(f(nv_array(1), TRUE), nv_array(2))
  expect_equal(f(nv_array(1), FALSE), nv_array(1))
})


test_that("jit: array return value is not wrapped in list", {
  f <- jit(nvl_add)
  out <- f(nv_scalar(1.2), nv_scalar(-0.7))
  expect_equal(as_array(out), 1.2 + (-0.7), tolerance = 1e-6)
})

test_that("constants can be part of the program", {
  f <- jit(function(x) x + nv_scalar(1))
  expect_equal(f(nv_array(1)), nv_array(2))
})

test_that("Only constants in group generics", {
  f <- jit(function() {
    nv_scalar(1) + nv_scalar(2)
    #nv_add(nv_scalar(1), nv_scalar(2))
  })
  expect_equal(f(), nv_scalar(3))
})

test_that("... works (#19)", {
  expect_equal(
    jit(sum)(nv_array(1:10)),
    nv_scalar(55L, dtype = "i32")
  )

  f <- function(..., a) {
    a + sum(...)
  }
  expect_equal(
    jit(f)(a = nv_scalar(1L), nv_array(1:10)),
    nv_scalar(56L, dtype = "i32")
  )
})

test_that("good error message when passing AbstractArrays", {
  expect_error(jit(nv_negate)(nv_abstract("f32", c(2, 2))), "autoconvert")
})

test_that("jit_eval does not modify calling environment", {
  x <- nv_array(1:2)
  jit_eval({
    x <- nv_array(3:4)
  })
  expect_equal(x, nv_array(1:2))
})

test_that("nested jit: jitted function can be called inside jit (#220)", {
  inner <- jit(function(x, y) x + y)
  outer <- jit(function(a, b) inner(a, b) * nv_scalar(2L))
  result <- outer(nv_array(3L), nv_array(4L))
  expect_equal(result, nv_array(14L))
})

test_that("hash for cache depends on in_tree (#122)", {
  f <- jit(
    \(...) {
      args <- list(...)
      args[[1]][[1L]][[1L]]
    }
  )
  expect_equal(cache_size(f), 0L)
  expect_equal(f(list(list(nv_scalar(1L)), nv_scalar(2L))), nv_scalar(1L))
  expect_equal(cache_size(f), 1L)
  expect_equal(f(list(list(nv_scalar(1L), nv_scalar(2L)))), nv_scalar(1L))
  expect_equal(cache_size(f), 2L)
})

describe("jit: backend and device combinations", {
  # Trivial function usable for any backend
  ident <- function(x) x

  it("backend = NULL, device = NULL uses default_backend()", {
    local_backend("xla")
    f <- jit(ident)
    expect_equal(backend(f), "xla")
  })

  it("backend = NULL, device = NULL follows default_backend() = 'quickr'", {
    skip_if_not_installed("quickr")
    local_backend("xla")
    f <- jit(ident)
    expect_equal(backend(f), "xla")
  })

  it("backend = 'xla', device = NULL uses xla", {
    f <- jit(ident, backend = "xla")
    expect_equal(backend(f), "xla")
  })

  it("backend = 'quickr', device = NULL uses quickr", {
    skip_if_not_installed("quickr")
    f <- jit(ident, backend = "quickr")
    expect_equal(backend(f), "quickr")
  })

  it("backend = NULL, device = PJRTDevice uses xla (derived from device)", {
    f <- jit(ident, device = pjrt::pjrt_device("cpu"))
    expect_equal(backend(f), "xla")
  })

  it("backend = NULL, device = QuickrDevice uses quickr (derived from device)", {
    skip_if_not_installed("quickr")
    f <- jit(ident, device = QuickrDevice("cpu"))
    expect_equal(backend(f), "quickr")
  })

  it("backend = 'xla', device = PJRTDevice is consistent", {
    f <- jit(ident, backend = "xla", device = pjrt::pjrt_device("cpu"))
    expect_equal(backend(f), "xla")
  })

  it("backend = 'quickr', device = QuickrDevice is consistent", {
    skip_if_not_installed("quickr")
    f <- jit(ident, backend = "quickr", device = QuickrDevice("cpu"))
    expect_equal(backend(f), "quickr")
  })

  it("backend = 'xla' conflicts with device = QuickrDevice", {
    skip_if_not_installed("quickr")
    expect_error(
      jit(ident, backend = "xla", device = QuickrDevice("cpu")),
      "has backend.*quickr.*backend.*xla"
    )
  })

  it("backend = 'quickr' conflicts with device = PJRTDevice", {
    skip_if_not_installed("quickr")
    expect_error(
      jit(ident, backend = "quickr", device = pjrt::pjrt_device("cpu")),
      "has backend.*xla.*backend.*quickr"
    )
  })

  it("backend = NULL, device = 'cpu' resolves via default_backend()", {
    withr::local_options(anvil.default_backend = "xla")
    f <- jit(ident, device = "cpu")
    expect_equal(backend(f), "xla")
  })

  it("backend = 'xla', device = 'cpu' resolves to xla", {
    f <- jit(ident, backend = "xla", device = "cpu")
    expect_equal(backend(f), "xla")
  })

  it("backend = 'quickr', device = 'cpu' resolves to quickr", {
    skip_if_not_installed("quickr")
    f <- jit(ident, backend = "quickr", device = "cpu")
    expect_equal(backend(f), "quickr")
  })

  it("backend = 'auto' routes to jit_auto (attr is 'auto' until called)", {
    f <- jit(ident, backend = "auto")
    expect_equal(backend(f), "auto")
    # At call time, backend is picked from the input.
    out <- f(nv_scalar(1))
    expect_equal(backend(out), "xla")
  })

  it("device = from_arg(...) routes to jit_auto", {
    g <- function(value, dev = NULL) nv_fill(value, shape = c(), dtype = "f32", device = dev)
    f <- jit(g, static = c("value", "dev"), device = from_arg("dev"))
    expect_equal(backend(f), "auto")
    out <- f(1, dev = pjrt::pjrt_device("cpu"))
    expect_equal(backend(out), "xla")
  })

  it("device = from_arg(...) with QuickrDevice routes to quickr", {
    skip_if_not_installed("quickr")
    g <- function(value, dev = NULL) nv_fill(value, shape = c(), dtype = "f32", device = dev)
    f <- jit(g, static = c("value", "dev"), device = from_arg("dev"))
    out <- f(1, dev = QuickrDevice("cpu"))
    expect_equal(backend(out), "quickr")
  })
})
