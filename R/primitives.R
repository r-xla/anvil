#' @include utils.R
#' @include type-converters.R

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
    env <- zero_env()
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
  graph_call(p_add, list(lhs, rhs))[[1L]]
}

p_mul <- Primitive("mul")
nvl_mul <- function(lhs, rhs) {
  graph_call(p_mul, list(lhs, rhs))[[1L]]
}

p_sub <- Primitive("sub")
nvl_sub <- function(lhs, rhs) {
  graph_call(p_sub, list(lhs, rhs))[[1L]]
}
p_neg <- Primitive("negate")
nvl_neg <- function(operand) {
  graph_call(p_neg, list(operand))[[1L]]
}
p_div <- Primitive("divide")
nvl_div <- function(lhs, rhs) {
  graph_call(p_div, list(lhs, rhs))[[1L]]
}
p_pow <- Primitive("power")
nvl_pow <- function(lhs, rhs) {
  graph_call(p_pow, list(lhs, rhs))[[1L]]
}

p_broadcast_in_dim <- Primitive("broadcast_in_dim")

nvl_broadcast_in_dim <- function(operand, shape_out, broadcast_dimensions) {
  graph_call(
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
  graph_call(
    p_dot_general,
    list(lhs, rhs),
    list(contracting_dims = contracting_dims, batching_dims = batching_dims)
  )[[1L]]
}

p_transpose <- Primitive("transpose")
nvl_transpose <- function(operand, permutation) {
  graph_call(
    p_transpose,
    list(operand),
    list(permutation = permutation)
  )[[1L]]
}

p_reshape <- Primitive("reshape")
nvl_reshape <- function(operand, shape) {
  graph_call(
    p_reshape,
    list(operand),
    params = list(shape = shape)
  )[[1L]]
}

# reduction operators

p_reduce_sum <- Primitive("sum")
nvl_reduce_sum <- function(operand, dims, drop = TRUE) {
  graph_call(
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
  graph_call(
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
  graph_call(
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
  graph_call(
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
  graph_call(
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
  graph_call(
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
  graph_call(p_eq, list(lhs, rhs))[[1L]]
}

p_ne <- Primitive("not_equal")
nvl_ne <- function(lhs, rhs) {
  graph_call(p_ne, list(lhs, rhs))[[1L]]
}

p_gt <- Primitive("greater")
nvl_gt <- function(lhs, rhs) {
  graph_call(p_gt, list(lhs, rhs))[[1L]]
}

p_ge <- Primitive("greater_equal")
nvl_ge <- function(lhs, rhs) {
  graph_call(p_ge, list(lhs, rhs))[[1L]]
}

p_lt <- Primitive("less")
nvl_lt <- function(lhs, rhs) {
  graph_call(p_lt, list(lhs, rhs))[[1L]]
}

p_le <- Primitive("less_equal")
nvl_le <- function(lhs, rhs) {
  graph_call(p_le, list(lhs, rhs))[[1L]]
}

# additional simple binary primitives -----------------------------------------

p_max <- Primitive("maximum")
nvl_max <- function(lhs, rhs) {
  graph_call(p_max, list(lhs, rhs))[[1L]]
}

p_min <- Primitive("minimum")
nvl_min <- function(lhs, rhs) {
  graph_call(p_min, list(lhs, rhs))[[1L]]
}

p_remainder <- Primitive("remainder")
nvl_remainder <- function(lhs, rhs) {
  graph_call(p_remainder, list(lhs, rhs))[[1L]]
}

p_and <- Primitive("and")
nvl_and <- function(lhs, rhs) {
  graph_call(p_and, list(lhs, rhs))[[1L]]
}

p_not <- Primitive("not")
nvl_not <- function(operand) {
  graph_call(p_not, list(operand))[[1L]]
}

p_or <- Primitive("or")
nvl_or <- function(lhs, rhs) {
  graph_call(p_or, list(lhs, rhs))[[1L]]
}

p_xor <- Primitive("xor")
nvl_xor <- function(lhs, rhs) {
  graph_call(p_xor, list(lhs, rhs))[[1L]]
}

p_shift_left <- Primitive("shift_left")
nvl_shift_left <- function(lhs, rhs) {
  graph_call(p_shift_left, list(lhs, rhs))[[1L]]
}

p_shift_right_logical <- Primitive("shift_right_logical")
nvl_shift_right_logical <- function(lhs, rhs) {
  graph_call(p_shift_right_logical, list(lhs, rhs))[[1L]]
}

p_shift_right_arithmetic <- Primitive("shift_right_arithmetic")
nvl_shift_right_arithmetic <- function(lhs, rhs) {
  graph_call(p_shift_right_arithmetic, list(lhs, rhs))[[1L]]
}

p_atan2 <- Primitive("atan2")
nvl_atan2 <- function(lhs, rhs) {
  graph_call(p_atan2, list(lhs, rhs))[[1L]]
}

# unary math primitives ---------------------------------------------------------

p_abs <- Primitive("abs")
nvl_abs <- function(operand) {
  graph_call(p_abs, list(operand))[[1L]]
}

p_sqrt <- Primitive("sqrt")
nvl_sqrt <- function(operand) {
  graph_call(p_sqrt, list(operand))[[1L]]
}

p_rsqrt <- Primitive("rsqrt")
nvl_rsqrt <- function(operand) {
  graph_call(p_rsqrt, list(operand))[[1L]]
}

p_log <- Primitive("log")
nvl_log <- function(operand) {
  graph_call(p_log, list(operand))[[1L]]
}

p_tanh <- Primitive("tanh")
nvl_tanh <- function(operand) {
  graph_call(p_tanh, list(operand))[[1L]]
}

p_tan <- Primitive("tan")
nvl_tan <- function(operand) {
  graph_call(p_tan, list(operand))[[1L]]
}

p_floor <- Primitive("floor")
nvl_floor <- function(operand) {
  graph_call(p_floor, list(operand))[[1L]]
}

p_ceil <- Primitive("ceil")
nvl_ceil <- function(operand) {
  graph_call(p_ceil, list(operand))[[1L]]
}

p_sign <- Primitive("sign")
nvl_sign <- function(operand) {
  graph_call(p_sign, list(operand))[[1L]]
}

p_exp <- Primitive("exp")
nvl_exp <- function(operand) {
  graph_call(p_exp, list(operand))[[1L]]
}

p_round <- Primitive("round")
nvl_round <- function(operand, method = "nearest_even") {
  if (!(method %in% c("nearest_even", "afz"))) {
    cli_abort("method must be one of: 'nearest_even', 'afz', but is {method}")
  }
  graph_call(p_round, list(operand), list(method = method))[[1L]]
}

# dtype conversion ----------------------------------------------------------------

p_convert <- Primitive("convert")
nvl_convert <- function(operand, dtype) {
  graph_call(
    p_convert,
    list(operand),
    params = list(dtype = dtype)
  )[[1L]]
}

# control flow primitives -------------------------------------------------------

p_select <- Primitive("select")
nvl_select <- function(pred, true_value, false_value) {
  graph_call(p_select, list(pred, true_value, false_value))[[1L]]
}

p_if <- Primitive("if")
nvl_if <- function(pred, true, false) {
  true_expr <- rlang::enquo(true)
  false_expr <- rlang::enquo(false)

  # Build sub-graphs for each branch (no inputs, just capture closed-over values)
  true_graph <- graphify(function() rlang::eval_tidy(true_expr), list())
  false_graph <- graphify(function() rlang::eval_tidy(false_expr), list())

  if (!identical(true_graph@out_tree, false_graph@out_tree)) {
    cli_abort("true and false branches must have the same output structure")
  }

  out <- graph_call(
    p_if,
    list(pred),
    params = list(true_graph = true_graph, false_graph = false_graph)
  )
  unflatten(true_graph@out_tree, out)
}

p_while <- Primitive("while")
nvl_while <- function(cond, body, init) {
  # TODO:
}
