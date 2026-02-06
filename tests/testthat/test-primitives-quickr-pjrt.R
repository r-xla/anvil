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
  vals <- generate_test_data(shape, dtype = dtype, non_negative = non_negative) # nolint
  if (!length(shape)) {
    return(vals[[1L]])
  }
  array(vals, dim = shape)
}

expect_quickr_matches_pjrt <- function(fn, templates, args, tolerance = 1e-12, info = NULL) {
  graph <- trace_fn(fn, templates)
  f_quick <- graph_to_quickr_function(graph)
  run_pjrt <- compile_graph_pjrt(graph) # nolint

  args <- args[names(templates)]
  out_quick <- do.call(f_quick, unname(args))
  out_pjrt <- do.call(run_pjrt, unname(args))
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
  withr::local_seed(seed)
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

compare_case <- function(op, name, seed) {
  withr::local_seed(seed)
  shape <- c(2L, 3L)
  x <- make_input(shape)
  y <- make_input(shape)
  templates <- list(
    x = make_template(shape),
    y = make_template(shape)
  )
  list(quickr_case(function(x, y) op(x, y), templates, list(x = x, y = y), info = name))
}

pred_binary_case <- function(op, name, seed) {
  withr::local_seed(seed)
  shape <- c(2L, 3L)
  x <- matrix(sample(c(TRUE, FALSE), 6, replace = TRUE), nrow = 2, ncol = 3)
  y <- matrix(sample(c(TRUE, FALSE), 6, replace = TRUE), nrow = 2, ncol = 3)
  templates <- list(
    x = make_template(shape, dtype = "pred"),
    y = make_template(shape, dtype = "pred")
  )
  list(quickr_case(function(x, y) op(x, y), templates, list(x = x, y = y), info = name))
}

quickr_pjrt_cases <- list(
  fill = function() {
    case <- function(value, shape, dtype, info) {
      quickr_case(
        function() nv_fill(value, shape = shape, dtype = dtype),
        list(),
        list(),
        info = info
      )
    }
    list(
      case(3.25, integer(), "f64", "fill: f64 scalar"),
      case(2L, c(4L), "i32", "fill: i32 vector"),
      case(TRUE, c(2L, 3L), "pred", "fill: pred matrix")
    )
  },
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
  convert = function() {
    withr::local_seed(12)

    x <- matrix(rnorm(6, sd = 0.2), nrow = 2, ncol = 3)
    y <- matrix(sample.int(10, 6, replace = TRUE), nrow = 2, ncol = 3)
    z <- matrix(sample(c(TRUE, FALSE), 6, replace = TRUE), nrow = 2, ncol = 3)

    list(
      quickr_case(
        function(x) nvl_convert(x, dtype = "i32"),
        list(x = make_template(dim(x), dtype = "f64")),
        list(x = x),
        info = "convert: f64 -> i32"
      ),
      quickr_case(
        function(y) nvl_convert(y, dtype = "f64"),
        list(y = make_template(dim(y), dtype = "i32")),
        list(y = y),
        info = "convert: i32 -> f64"
      ),
      quickr_case(
        function(z) nvl_convert(z, dtype = "pred"),
        list(z = make_template(dim(z), dtype = "pred")),
        list(z = z),
        info = "convert: pred -> pred"
      )
    )
  },
  negate = function() {
    withr::local_seed(5)
    shape <- c(2L, 3L)
    x <- make_input(shape)
    list(quickr_case(function(x) nv_negate(x), list(x = make_template(shape)), list(x = x), info = "negate"))
  },
  reshape = function() {
    withr::local_seed(6)
    shape_in <- c(2L, 3L)
    shape_out <- c(3L, 2L)
    x <- make_input(shape_in)
    list(quickr_case(
      function(x) nvl_reshape(x, shape_out),
      list(x = make_template(shape_in)),
      list(x = x),
      info = "reshape"
    ))
  },
  transpose = function() {
    withr::local_seed(7)
    shape <- c(2L, 3L)
    x <- make_input(shape)
    list(quickr_case(
      function(x) nvl_transpose(x, permutation = c(2L, 1L)),
      list(x = make_template(shape)),
      list(x = x),
      info = "transpose"
    ))
  },
  broadcast_in_dim = function() {
    withr::local_seed(8)
    shape_in <- c(2L, 1L)
    shape_out <- c(2L, 3L)
    x <- make_input(shape_in)
    list(quickr_case(
      function(x) nvl_broadcast_in_dim(x, shape = shape_out, broadcast_dimensions = c(1L, 2L)),
      list(x = make_template(shape_in)),
      list(x = x),
      info = "broadcast_in_dim"
    ))
  },
  dot_general = function() {
    withr::local_seed(9)
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
    withr::local_seed(10)
    shape <- c(2L, 3L)
    x <- make_input(shape)
    list(quickr_case(function(x) sum(x), list(x = make_template(shape)), list(x = x), info = "sum"))
  },
  reduce_sum = function() {
    withr::local_seed(11)
    shape <- c(2L, 3L)
    x <- make_input(shape)
    list(quickr_case(
      function(x) nvl_reduce_sum(x, dims = 1L, drop = TRUE),
      list(x = make_template(shape)),
      list(x = x),
      info = "reduce_sum"
    ))
  },
  select = function() {
    withr::local_seed(13)

    shape <- c(2L, 3L)
    pred <- matrix(sample(c(TRUE, FALSE), 6, replace = TRUE), nrow = 2, ncol = 3)
    x <- make_input(shape)
    y <- make_input(shape)

    list(quickr_case(
      function(pred, x, y) nvl_ifelse(pred, x, y),
      list(
        pred = make_template(shape, dtype = "pred"),
        x = make_template(shape, dtype = "f64"),
        y = make_template(shape, dtype = "f64")
      ),
      list(pred = pred, x = x, y = y),
      info = "select"
    ))
  },
  equal = function() {
    compare_case(nv_eq, "eq", seed = 14)
  },
  not_equal = function() {
    compare_case(nv_ne, "ne", seed = 15)
  },
  greater = function() {
    compare_case(nv_gt, "gt", seed = 16)
  },
  greater_equal = function() {
    compare_case(nv_ge, "ge", seed = 17)
  },
  less = function() {
    compare_case(nv_lt, "lt", seed = 18)
  },
  less_equal = function() {
    compare_case(nv_le, "le", seed = 19)
  },
  and = function() {
    pred_binary_case(nv_and, "and", seed = 20)
  },
  or = function() {
    pred_binary_case(nv_or, "or", seed = 21)
  },
  xor = function() {
    pred_binary_case(nv_xor, "xor", seed = 22)
  },
  not = function() {
    withr::local_seed(23)
    shape <- c(2L, 3L)
    x <- matrix(sample(c(TRUE, FALSE), 6, replace = TRUE), nrow = 2, ncol = 3)
    list(quickr_case(
      function(x) nv_not(x),
      list(x = make_template(shape, dtype = "pred")),
      list(x = x),
      info = "not"
    ))
  }
)

quickr_primitives <- sort(c(
  "fill",
  "convert",
  "add",
  "sub",
  "mul",
  "divide",
  "negate",
  "equal",
  "not_equal",
  "greater",
  "greater_equal",
  "less",
  "less_equal",
  "and",
  "or",
  "xor",
  "not",
  "select",
  "broadcast_in_dim",
  "dot_general",
  "transpose",
  "reshape",
  "sum",
  "reduce_sum"
))

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
