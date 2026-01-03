skip_quickr_pjrt <- function() {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")
}

make_template <- function(shape, dtype = "f64") {
  if (!length(shape)) {
    return(nv_scalar(0, dtype = dtype))
  }
  nv_tensor(array(0, dim = shape), dtype = dtype, shape = shape)
}

make_input <- function(shape, dtype = "f64", non_negative = FALSE) {
  vals <- generate_test_data(shape, dtype = dtype, non_negative = non_negative)
  if (!length(shape)) {
    return(vals[[1L]])
  }
  array(vals, dim = shape)
}

expect_quickr_matches_pjrt <- function(fn, templates, args, tolerance = 1e-12, info = NULL) {
  graph <- trace_fn(fn, templates)
  f_quick <- graph_to_quickr_function(graph)
  run_pjrt <- compile_graph_pjrt(graph)

  out_quick <- rlang::exec(f_quick, !!!args)
  out_pjrt <- rlang::exec(run_pjrt, !!!args)
  out_quick <- normalize_quickr_output(out_quick, out_pjrt)

  testthat::expect_equal(out_quick, out_pjrt, tolerance = tolerance, info = info)
}

normalize_quickr_output <- function(x, ref) {
  if (is.list(x) && is.list(ref)) {
    return(Map(normalize_quickr_output, x, ref))
  }
  if (is.atomic(x) && is.atomic(ref)) {
    if (is.null(dim(x)) && !is.null(dim(ref))) {
      dim(x) <- dim(ref)
    }
  }
  x
}

test_that("quickr matches pjrt for elementwise binary ops", {
  skip_quickr_pjrt()
  set.seed(1)

  shape <- c(2L, 3L)
  x <- make_input(shape)
  y <- make_input(shape)

  templates <- list(
    x = make_template(shape),
    y = make_template(shape)
  )

  ops <- list(
    add = nv_add,
    sub = nv_sub,
    mul = nv_mul,
    div = nv_div
  )

  for (name in names(ops)) {
    op <- ops[[name]]
    y_use <- if (name == "div") y + 1 else y
    expect_quickr_matches_pjrt(
      function(x, y) op(x, y),
      templates,
      args = list(x = x, y = y_use),
      info = name
    )
  }
})

test_that("quickr matches pjrt for unary negate", {
  skip_quickr_pjrt()
  set.seed(2)

  shape <- c(2L, 3L)
  x <- make_input(shape)

  expect_quickr_matches_pjrt(
    function(x) nv_neg(x),
    list(x = make_template(shape)),
    args = list(x = x),
    info = "negate"
  )
})

test_that("quickr matches pjrt for reshape", {
  skip_quickr_pjrt()
  set.seed(3)

  shape_in <- c(2L, 3L)
  shape_out <- c(3L, 2L)
  x <- make_input(shape_in)

  expect_quickr_matches_pjrt(
    function(x) nvl_reshape(x, shape_out),
    list(x = make_template(shape_in)),
    args = list(x = x),
    info = "reshape"
  )
})

test_that("quickr matches pjrt for transpose", {
  skip_quickr_pjrt()
  set.seed(4)

  shape <- c(2L, 3L)
  x <- make_input(shape)

  expect_quickr_matches_pjrt(
    function(x) nvl_transpose(x, permutation = c(2L, 1L)),
    list(x = make_template(shape)),
    args = list(x = x),
    info = "transpose"
  )
})

test_that("quickr matches pjrt for broadcast_in_dim", {
  skip_quickr_pjrt()
  set.seed(5)

  shape_in <- c(2L, 1L)
  shape_out <- c(2L, 3L)
  x <- make_input(shape_in)

  expect_quickr_matches_pjrt(
    function(x) nvl_broadcast_in_dim(x, shape_out = shape_out, broadcast_dimensions = c(1L, 2L)),
    list(x = make_template(shape_in)),
    args = list(x = x),
    info = "broadcast_in_dim"
  )
})

test_that("quickr matches pjrt for dot_general", {
  skip_quickr_pjrt()
  set.seed(6)

  lhs_shape <- c(2L, 3L)
  rhs_shape <- c(3L, 4L)
  lhs <- make_input(lhs_shape)
  rhs <- make_input(rhs_shape)

  expect_quickr_matches_pjrt(
    function(lhs, rhs) {
      nvl_dot_general(
        lhs,
        rhs,
        contracting_dims = list(2L, 1L),
        batching_dims = list(integer(), integer())
      )
    },
    list(
      lhs = make_template(lhs_shape),
      rhs = make_template(rhs_shape)
    ),
    args = list(lhs = lhs, rhs = rhs),
    info = "dot_general"
  )
})

test_that("quickr matches pjrt for reduce_sum", {
  skip_quickr_pjrt()
  set.seed(7)

  shape <- c(2L, 3L)
  x <- make_input(shape)

  expect_quickr_matches_pjrt(
    function(x) nvl_reduce_sum(x, dims = 1L, drop = TRUE),
    list(x = make_template(shape)),
    args = list(x = x),
    info = "reduce_sum"
  )
})
