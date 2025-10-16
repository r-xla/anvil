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
    rules = S7::class_environment
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
  rules <- sapply(globals$interpretation_rules, \(rule) {
    if (!is.null(x@rules[[rule]])) rule
  })
  rules_str <- if (length(rules) > 0L) paste0(rules, collapse = ", ") else "-"
  cat(" implements:", rules_str, "\n")
}

# We define these infix operators, because `+` etc. are reserved for the broadcasted ones (nv_)

`%+%` <- function(lhs, rhs) {
  nvl_add(lhs, rhs)
}

`%x%` <- function(lhs, rhs) {
  nvl_mul(lhs, rhs)
}

`%-%` <- function(lhs, rhs) {
  nvl_sub(lhs, rhs)
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

# transpose
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
