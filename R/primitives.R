#' @title Primitive
#' @description
#' Primitive interpretation rule.
#' @param name (`character()`)\cr
#'   The name of the primitive.
#' @return (`Primitive`)
#' @export
Primitive <- new_class(
  "Primitive",
  properties = list(
    name = class_character,
    rules = class_environment
  ),
  constructor = function(name) {
    env <- new.env(parent = emptyenv())
    new_object(S7_object(), rules = env, name = name)
  }
)


#' @export
`[[<-.anvil::Primitive` <- function(x, name, value) {
  if (!is.function(value)) {
    cli_abort("Rule must be a function")
  }
  x@rules[[name]] <- value
  if (!(name %in% globals$interpretation_rules)) {
    cli_abort("Unknown interpretation rule: {name}")
  }
  x
}

method(`[[`, Primitive) <- function(x, name) {
  rule <- x@rules[[name]]
  if (is.null(rule)) {
    if (!(name %in% globals$interpretation_rules)) {
      cli_abort("Unknown rule: {name}")
    }
    cli_abort("Rule {name} not defined for primitive {x@name}")
  }
  rule
}

method(print, Primitive) <- function(x, ...) {
  cat(sprintf("<Primitive:%s>\n", x@name))
}

p_add <- Primitive("add")
nvl_add <- function(lhs, rhs) {
  interprete(p_add, list(lhs, rhs))[[1L]]
}

p_mul <- Primitive("mul")
nvl_mul <- function(lhs, rhs) {
  interprete(p_mul, list(lhs, rhs))[[1L]]
}

p_sub <- Primitive("sub")
nvl_sub <- function(lhs, rhs) {
  interprete(p_sub, list(lhs, rhs))[[1L]]
}
p_neg <- Primitive("negate")
nvl_neg <- function(operand) {
  interprete(p_neg, list(operand))[[1L]]
}
p_div <- Primitive("divide")
nvl_div <- function(lhs, rhs) {
  interprete(p_div, list(lhs, rhs))[[1L]]
}
p_pow <- Primitive("power")
nvl_pow <- function(lhs, rhs) {
  interprete(p_pow, list(lhs, rhs))[[1L]]
}

p_broadcast_in_dim <- Primitive("broadcast_in_dim")

nvl_broadcast_in_dim <- function(operand, shape_out, broadcast_dimensions) {
  interprete(
    p_broadcast_in_dim,
    list(operand),
    params = list(
      shape_out = shape_out,
      broadcast_dimensions = broadcast_dimensions
    )
  )[[1L]]
}

p_dot_general <- Primitive("dot_general")
nvl_dot_general <- function(lhs, rhs, contracting_dims, batching_dims) {
  interprete(
    p_dot_general,
    list(lhs, rhs),
    list(contracting_dims = contracting_dims, batching_dims = batching_dims)
  )[[1L]]
}

p_transpose <- Primitive("transpose")
nvl_transpose <- function(operand, permutation) {
  interprete(
    p_transpose,
    list(operand),
    list(permutation = permutation)
  )[[1L]]
}

p_reshape <- Primitive("reshape")
nvl_reshape <- function(operand, shape) {
  interprete(
    p_reshape,
    list(operand),
    params = list(shape = shape)
  )[[1L]]
}

# reduction operators

p_reduce_sum <- Primitive("sum")
nvl_reduce_sum <- function(operand, dims, drop = TRUE) {
  interprete(
    p_reduce_sum,
    list(operand),
    params = list(
      dims = dims,
      drop = drop
    )
  )[[1L]]
}

p_reduce_prod <- Primitive("prod")
nvl_reduce_prod <- function(operand, dims, drop = TRUE) {
  interprete(
    p_reduce_prod,
    list(operand),
    params = list(
      dims = dims,
      drop = drop
    )
  )[[1L]]
}

p_reduce_max <- Primitive("max")
nvl_reduce_max <- function(operand, dims, drop = TRUE) {
  interprete(
    p_reduce_max,
    list(operand),
    params = list(
      dims = dims,
      drop = drop
    )
  )[[1L]]
}

p_reduce_min <- Primitive("min")
nvl_reduce_min <- function(operand, dims, drop = TRUE) {
  interprete(
    p_reduce_min,
    list(operand),
    params = list(
      dims = dims,
      drop = drop
    )
  )[[1L]]
}

p_reduce_any <- Primitive("any")
nvl_reduce_any <- function(operand, dims, drop = TRUE) {
  interprete(
    p_reduce_any,
    list(operand),
    params = list(
      dims = dims,
      drop = drop
    )
  )[[1L]]
}

p_reduce_all <- Primitive("all")
nvl_reduce_all <- function(operand, dims, drop = TRUE) {
  interprete(
    p_reduce_all,
    list(operand),
    params = list(
      dims = dims,
      drop = drop
    )
  )[[1L]]
}

# comparison primitives --------------------------------------------------------

p_eq <- Primitive("equal")
nvl_eq <- function(lhs, rhs) {
  interprete(p_eq, list(lhs, rhs))[[1L]]
}

p_ne <- Primitive("not_equal")
nvl_ne <- function(lhs, rhs) {
  interprete(p_ne, list(lhs, rhs))[[1L]]
}

p_gt <- Primitive("greater")
nvl_gt <- function(lhs, rhs) {
  interprete(p_gt, list(lhs, rhs))[[1L]]
}

p_ge <- Primitive("greater_equal")
nvl_ge <- function(lhs, rhs) {
  interprete(p_ge, list(lhs, rhs))[[1L]]
}

p_lt <- Primitive("less")
nvl_lt <- function(lhs, rhs) {
  interprete(p_lt, list(lhs, rhs))[[1L]]
}

p_le <- Primitive("less_equal")
nvl_le <- function(lhs, rhs) {
  interprete(p_le, list(lhs, rhs))[[1L]]
}

# additional simple binary primitives -----------------------------------------

p_max <- Primitive("maximum")
nvl_max <- function(lhs, rhs) {
  interprete(p_max, list(lhs, rhs))[[1L]]
}

p_min <- Primitive("minimum")
nvl_min <- function(lhs, rhs) {
  interprete(p_min, list(lhs, rhs))[[1L]]
}

p_remainder <- Primitive("remainder")
nvl_remainder <- function(lhs, rhs) {
  interprete(p_remainder, list(lhs, rhs))[[1L]]
}

p_and <- Primitive("and")
nvl_and <- function(lhs, rhs) {
  interprete(p_and, list(lhs, rhs))[[1L]]
}

p_not <- Primitive("not")
nvl_not <- function(operand) {
  interprete(p_not, list(operand))[[1L]]
}

p_or <- Primitive("or")
nvl_or <- function(lhs, rhs) {
  interprete(p_or, list(lhs, rhs))[[1L]]
}

p_xor <- Primitive("xor")
nvl_xor <- function(lhs, rhs) {
  interprete(p_xor, list(lhs, rhs))[[1L]]
}

p_shift_left <- Primitive("shift_left")
nvl_shift_left <- function(lhs, rhs) {
  interprete(p_shift_left, list(lhs, rhs))[[1L]]
}

p_shift_right_logical <- Primitive("shift_right_logical")
nvl_shift_right_logical <- function(lhs, rhs) {
  interprete(p_shift_right_logical, list(lhs, rhs))[[1L]]
}

p_shift_right_arithmetic <- Primitive("shift_right_arithmetic")
nvl_shift_right_arithmetic <- function(lhs, rhs) {
  interprete(p_shift_right_arithmetic, list(lhs, rhs))[[1L]]
}

p_atan2 <- Primitive("atan2")
nvl_atan2 <- function(lhs, rhs) {
  interprete(p_atan2, list(lhs, rhs))[[1L]]
}

# unary math primitives ---------------------------------------------------------

p_abs <- Primitive("abs")
nvl_abs <- function(operand) {
  interprete(p_abs, list(operand))[[1L]]
}

p_sqrt <- Primitive("sqrt")
nvl_sqrt <- function(operand) {
  interprete(p_sqrt, list(operand))[[1L]]
}

p_rsqrt <- Primitive("rsqrt")
nvl_rsqrt <- function(operand) {
  interprete(p_rsqrt, list(operand))[[1L]]
}

p_log <- Primitive("log")
nvl_log <- function(operand) {
  interprete(p_log, list(operand))[[1L]]
}

p_tanh <- Primitive("tanh")
nvl_tanh <- function(operand) {
  interprete(p_tanh, list(operand))[[1L]]
}

p_tan <- Primitive("tan")
nvl_tan <- function(operand) {
  interprete(p_tan, list(operand))[[1L]]
}

p_floor <- Primitive("floor")
nvl_floor <- function(operand) {
  interprete(p_floor, list(operand))[[1L]]
}

p_ceil <- Primitive("ceil")
nvl_ceil <- function(operand) {
  interprete(p_ceil, list(operand))[[1L]]
}

p_sign <- Primitive("sign")
nvl_sign <- function(operand) {
  interprete(p_sign, list(operand))[[1L]]
}

p_exp <- Primitive("exp")
nvl_exp <- function(operand) {
  interprete(p_exp, list(operand))[[1L]]
}

p_round <- Primitive("round")
nvl_round <- function(operand, method = "nearest_even") {
  if (!(method %in% c("nearest_even", "afz"))) {
    cli_abort("method must be one of: 'nearest_even', 'afz', but is {method}")
  }
  interprete(p_round, list(operand), list(method = method))[[1L]]
}

# dtype conversion ----------------------------------------------------------------

p_convert <- Primitive("convert")
nvl_convert <- function(operand, dtype) {
  interprete(
    p_convert,
    list(operand),
    params = list(dtype = dtype)
  )[[1L]]
}

# control flow primitives -------------------------------------------------------

p_select <- Primitive("select")
nvl_select <- function(pred, true_value, false_value) {
  interprete(p_select, list(pred, true_value, false_value))[[1L]]
}

p_if <- Primitive("if")
nvl_if <- function(pred, true, false) {
  true_expr <- rlang::enquo(true)
  false_expr <- rlang::enquo(false)
  true_fn <- flatten_fun(function() {
    rlang::eval_tidy(true_expr)
  })
  false_fn <- flatten_fun(function() {
    rlang::eval_tidy(false_expr)
  })

  true_out <- stablehlo(true_fn, list())
  false_out <- stablehlo(false_fn, list())

  if (!identical(true_out[[2L]], false_out[[2L]])) {
    cli_abort("true and false must have the same output tree")
  }
  out <- interprete(p_if, list(pred), params = list(true = true_out[[1L]], false = false_out[[1L]]))
  unflatten(true_out[[2L]], out)
}

p_while <- Primitive("while")
nvl_while <- function(cond, body, init) {
  # TODO:
}
