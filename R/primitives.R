# Core Primitives

Primitive <- new_class(
  "Primitive",
  properties = list(
    name = class_character,
    .rules = new_property(class_environment),
    jit_rule = S7::new_property(
      class_function,
      setter = function(self, value) {
        self@.rules[["jit_rule"]] <- value
        self
      },
      getter = function(self) {
        self@.rules[["jit_rule"]]
      }
    ),
    pullback_rule = S7::new_property(
      class_function,
      setter = function(self, value) {
        self@.rules[["pullback_rule"]] <- value
        self
      },
      getter = function(self) {
        self@.rules[["pullback_rule"]]
      }
    )
  ),
  constructor = function(name) {
    new_object(S7_object(), .rules = new.env(), name = name)
  }
)

method(print, Primitive) <- function(x, ...) {
  cat(sprintf("<Primitive:%s>\n", x@name))
}

register_jit_rule <- function(primitive, rule) {
  primitive@jit_rule <- rule
  primitive
}

register_pullback_rule <- function(primitive, rule) {
  primitive@pullback_rule <- rule
  primitive
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
