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

quickr_case <- function(fn, templates, args, tolerance = 1e-12, info = NULL) {
  list(
    fn = fn,
    templates = templates,
    args = args,
    tolerance = tolerance,
    info = info
  )
}

run_quickr_cases <- function(cases) {
  for (case in cases) {
    tol <- case$tolerance
    if (is.null(tol)) {
      tol <- 1e-12
    }
    expect_quickr_matches_pjrt(
      case$fn,
      case$templates,
      case$args,
      tolerance = tol,
      info = case$info
    )
  }
}

binary_case <- function(op, name, seed, adjust_y = NULL) {
  set.seed(seed)
  shape <- c(2L, 3L)
  x <- make_input(shape)
  y <- make_input(shape)
  if (!is.null(adjust_y)) {
    y <- adjust_y(y)
  }
  templates <- list(
    x = make_template(shape),
    y = make_template(shape)
  )
  list(quickr_case(function(x, y) op(x, y), templates, list(x = x, y = y), info = name))
}

quickr_pjrt_cases <- list(
  add = function() {
    binary_case(nv_add, "add", seed = 1)
  },
  sub = function() {
    binary_case(nv_sub, "sub", seed = 2)
  },
  mul = function() {
    binary_case(nv_mul, "mul", seed = 3)
  },
  divide = function() {
    binary_case(nv_div, "divide", seed = 4, adjust_y = function(y) y + 1)
  },
  negate = function() {
    set.seed(5)
    shape <- c(2L, 3L)
    x <- make_input(shape)
    list(quickr_case(function(x) nv_neg(x), list(x = make_template(shape)), list(x = x), info = "negate"))
  },
  reshape = function() {
    set.seed(6)
    shape_in <- c(2L, 3L)
    shape_out <- c(3L, 2L)
    x <- make_input(shape_in)
    list(quickr_case(function(x) nvl_reshape(x, shape_out), list(x = make_template(shape_in)), list(x = x), info = "reshape"))
  },
  transpose = function() {
    set.seed(7)
    shape <- c(2L, 3L)
    x <- make_input(shape)
    list(quickr_case(function(x) nvl_transpose(x, permutation = c(2L, 1L)), list(x = make_template(shape)), list(x = x), info = "transpose"))
  },
  broadcast_in_dim = function() {
    set.seed(8)
    shape_in <- c(2L, 1L)
    shape_out <- c(2L, 3L)
    x <- make_input(shape_in)
    list(quickr_case(
      function(x) nvl_broadcast_in_dim(x, shape_out = shape_out, broadcast_dimensions = c(1L, 2L)),
      list(x = make_template(shape_in)),
      list(x = x),
      info = "broadcast_in_dim"
    ))
  },
  dot_general = function() {
    set.seed(9)
    lhs_shape <- c(2L, 3L)
    rhs_shape <- c(3L, 4L)
    lhs <- make_input(lhs_shape)
    rhs <- make_input(rhs_shape)
    list(quickr_case(
      function(lhs, rhs) {
        nvl_dot_general(
          lhs,
          rhs,
          contracting_dims = list(2L, 1L),
          batching_dims = list(integer(), integer())
        )
      },
      list(lhs = make_template(lhs_shape), rhs = make_template(rhs_shape)),
      list(lhs = lhs, rhs = rhs),
      info = "dot_general"
    ))
  },
  sum = function() {
    set.seed(10)
    shape <- c(2L, 3L)
    x <- make_input(shape)
    list(quickr_case(function(x) sum(x), list(x = make_template(shape)), list(x = x), info = "sum"))
  },
  reduce_sum = function() {
    set.seed(11)
    shape <- c(2L, 3L)
    x <- make_input(shape)
    list(quickr_case(function(x) nvl_reduce_sum(x, dims = 1L, drop = TRUE), list(x = make_template(shape)), list(x = x), info = "reduce_sum"))
  }
)

quickr_primitives <- sort(ls(anvil:::quickr_lower_registry, all.names = TRUE))

for (prim in quickr_primitives) {
  test_that(paste0("quickr matches pjrt for primitive: ", prim), {
    skip_quickr_pjrt()
    case_fn <- quickr_pjrt_cases[[prim]]
    if (is.null(case_fn)) {
      testthat::skip(paste0("no quickr/PJRT parity case for primitive: ", prim))
    }
    cases <- case_fn()
    run_quickr_cases(cases)
  })
}
