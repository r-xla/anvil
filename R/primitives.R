#' @include utils.R
#' @include type-converters.R

# Type inference helper functions for graph building
#' @importFrom stablehlo infer_types_generic_biv infer_types_generic_uni
#' @importFrom stablehlo infer_types_boolean_biv infer_types_boolean_uni
#' @importFrom stablehlo infer_types_compare infer_types_transpose infer_types_reshape
#' @importFrom stablehlo infer_types_broadcast_in_dim infer_types_convert
#' @importFrom stablehlo infer_types_dot_general infer_types_select

infer_binary <- function(lhs, rhs) {
  stablehlo::infer_types_generic_biv(lhs, rhs)@items
}

infer_unary <- function(operand) {
  stablehlo::infer_types_generic_uni(operand)@items
}

infer_binary_boolean <- function(lhs, rhs) {
  stablehlo::infer_types_boolean_biv(lhs, rhs)@items
}

infer_unary_boolean <- function(operand) {
  stablehlo::infer_types_boolean_uni(operand)@items
}

infer_reduce <- function(operand, dims, drop) {
  old_shape <- operand@type@shape@dims
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(stablehlo::ValueType(stablehlo::TensorType(
    operand@type@dtype,
    stablehlo::Shape(new_shape)
  )))
}

infer_reduce_boolean <- function(operand, dims, drop) {
  old_shape <- operand@type@shape@dims
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(stablehlo::ValueType(stablehlo::TensorType(
    stablehlo::BooleanType(),
    stablehlo::Shape(new_shape)
  )))
}

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

HigherOrderPrimitive <- new_class(
  "HigherOrderPrimitive",
  parent = Primitive
)

is_higher_order_primitive <- function(x) {
  inherits(x, "anvil::HigherOrderPrimitive")
}


#' @export
`[[<-.anvil::Primitive` <- function(x, name, value) {
  if (!is.function(value)) {
    cli_abort("Rule must be a function")
  }
  x@rules[[name]] <- value
  if (!(name %in% globals$interpretation_rules)) {
    cli_abort("Unknown interpretation rule: {.val {name}}")
  }
  x
}

method(`[[`, Primitive) <- function(x, name) {
  rule <- x@rules[[name]]
  if (is.null(rule)) {
    if (!(name %in% globals$interpretation_rules)) {
      cli_abort("Unknown rule: {name}")
    }
    cli_abort("Rule {.field {name}} not defined for primitive {.field {x@name}}")
  }
  rule
}

method(print, Primitive) <- function(x, ...) {
  cat(sprintf("<Primitive:%s>\n", x@name))
}

p_add <- Primitive("add")
nvl_add <- function(lhs, rhs) {
  graph_desc_add(p_add, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
}

p_mul <- Primitive("mul")
nvl_mul <- function(lhs, rhs) {
  graph_desc_add(p_mul, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
}

p_sub <- Primitive("sub")
nvl_sub <- function(lhs, rhs) {
  graph_desc_add(p_sub, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
}
p_neg <- Primitive("negate")
nvl_neg <- function(operand) {
  graph_desc_add(p_neg, list(operand), infer_fn = infer_unary)[[1L]]
}
p_div <- Primitive("divide")
nvl_div <- function(lhs, rhs) {
  graph_desc_add(p_div, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
}
p_pow <- Primitive("power")
nvl_pow <- function(lhs, rhs) {
  graph_desc_add(p_pow, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
}

p_broadcast_in_dim <- Primitive("broadcast_in_dim")

nvl_broadcast_in_dim <- function(operand, shape_out, broadcast_dimensions) {
  infer_fn <- function(operand, shape_out, broadcast_dimensions) {
    bd_attr <- stablehlo::r_to_constant(
      as.integer(broadcast_dimensions - 1L),
      dtype = "i64",
      shape = length(broadcast_dimensions)
    )
    stablehlo::infer_types_broadcast_in_dim(
      operand,
      broadcast_dimensions = bd_attr,
      shape_out = shape_out
    )@items
  }
  graph_desc_add(
    p_broadcast_in_dim,
    list(operand),
    params = list(
      shape_out = shape_out,
      broadcast_dimensions = broadcast_dimensions
    ),
    infer_fn = infer_fn
  )[[1L]]
}

p_dot_general <- Primitive("dot_general")
nvl_dot_general <- function(lhs, rhs, contracting_dims, batching_dims) {
  infer_fn <- function(lhs, rhs, contracting_dims, batching_dims) {
    ddn <- stablehlo::DotDimensionNumbers(
      contracting_dims = lapply(contracting_dims, \(x) x - 1L),
      batching_dims = lapply(batching_dims, \(x) x - 1L)
    )
    stablehlo::infer_types_dot_general(lhs, rhs, dot_dimension_numbers = ddn)@items
  }
  graph_desc_add(
    p_dot_general,
    list(lhs, rhs),
    list(contracting_dims = contracting_dims, batching_dims = batching_dims),
    infer_fn = infer_fn
  )[[1L]]
}

p_transpose <- Primitive("transpose")
nvl_transpose <- function(operand, permutation) {
  infer_fn <- function(operand, permutation) {
    perm_attr <- stablehlo::r_to_constant(
      as.integer(permutation - 1L),
      dtype = "i64",
      shape = length(permutation)
    )
    stablehlo::infer_types_transpose(operand, permutation = perm_attr)@items
  }
  graph_desc_add(
    p_transpose,
    list(operand),
    list(permutation = permutation),
    infer_fn = infer_fn
  )[[1L]]
}

p_reshape <- Primitive("reshape")
nvl_reshape <- function(operand, shape) {
  infer_fn <- function(operand, shape) {
    stablehlo::infer_types_reshape(operand, shape_out = shape)@items
  }
  graph_desc_add(
    p_reshape,
    list(operand),
    params = list(shape = shape),
    infer_fn = infer_fn
  )[[1L]]
}

# reduction operators

p_reduce_sum <- Primitive("sum")
nvl_reduce_sum <- function(operand, dims, drop = TRUE) {
  graph_desc_add(
    p_reduce_sum,
    list(operand),
    params = list(dims = dims, drop = drop),
    infer_fn = infer_reduce
  )[[1L]]
}

p_reduce_prod <- Primitive("prod")
nvl_reduce_prod <- function(operand, dims, drop = TRUE) {
  graph_desc_add(
    p_reduce_prod,
    list(operand),
    params = list(dims = dims, drop = drop),
    infer_fn = infer_reduce
  )[[1L]]
}

p_reduce_max <- Primitive("max")
nvl_reduce_max <- function(operand, dims, drop = TRUE) {
  graph_desc_add(
    p_reduce_max,
    list(operand),
    params = list(dims = dims, drop = drop),
    infer_fn = infer_reduce
  )[[1L]]
}

p_reduce_min <- Primitive("min")
nvl_reduce_min <- function(operand, dims, drop = TRUE) {
  graph_desc_add(
    p_reduce_min,
    list(operand),
    params = list(dims = dims, drop = drop),
    infer_fn = infer_reduce
  )[[1L]]
}

p_reduce_any <- Primitive("any")
nvl_reduce_any <- function(operand, dims, drop = TRUE) {
  graph_desc_add(
    p_reduce_any,
    list(operand),
    params = list(dims = dims, drop = drop),
    infer_fn = infer_reduce_boolean
  )[[1L]]
}

p_reduce_all <- Primitive("all")
nvl_reduce_all <- function(operand, dims, drop = TRUE) {
  graph_desc_add(
    p_reduce_all,
    list(operand),
    params = list(dims = dims, drop = drop),
    infer_fn = infer_reduce_boolean
  )[[1L]]
}

# comparison primitives --------------------------------------------------------

p_eq <- Primitive("equal")
nvl_eq <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) {
    stablehlo::infer_types_compare(lhs, rhs, "EQ", "FLOAT")@items
  }
  graph_desc_add(p_eq, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_ne <- Primitive("not_equal")
nvl_ne <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) {
    stablehlo::infer_types_compare(lhs, rhs, "NE", "FLOAT")@items
  }
  graph_desc_add(p_ne, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_gt <- Primitive("greater")
nvl_gt <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) {
    stablehlo::infer_types_compare(lhs, rhs, "GT", "FLOAT")@items
  }
  graph_desc_add(p_gt, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_ge <- Primitive("greater_equal")
nvl_ge <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) {
    stablehlo::infer_types_compare(lhs, rhs, "GE", "FLOAT")@items
  }
  graph_desc_add(p_ge, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_lt <- Primitive("less")
nvl_lt <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) {
    stablehlo::infer_types_compare(lhs, rhs, "LT", "FLOAT")@items
  }
  graph_desc_add(p_lt, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_le <- Primitive("less_equal")
nvl_le <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) {
    stablehlo::infer_types_compare(lhs, rhs, "LE", "FLOAT")@items
  }
  graph_desc_add(p_le, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

# additional simple binary primitives -----------------------------------------

p_max <- Primitive("maximum")
nvl_max <- function(lhs, rhs) {
  graph_desc_add(p_max, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
}

p_min <- Primitive("minimum")
nvl_min <- function(lhs, rhs) {
  graph_desc_add(p_min, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
}

p_remainder <- Primitive("remainder")
nvl_remainder <- function(lhs, rhs) {
  graph_desc_add(p_remainder, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
}

p_and <- Primitive("and")
nvl_and <- function(lhs, rhs) {
  graph_desc_add(p_and, list(lhs, rhs), infer_fn = infer_binary_boolean)[[1L]]
}

p_not <- Primitive("not")
nvl_not <- function(operand) {
  graph_desc_add(p_not, list(operand), infer_fn = infer_unary_boolean)[[1L]]
}

p_or <- Primitive("or")
nvl_or <- function(lhs, rhs) {
  graph_desc_add(p_or, list(lhs, rhs), infer_fn = infer_binary_boolean)[[1L]]
}

p_xor <- Primitive("xor")
nvl_xor <- function(lhs, rhs) {
  graph_desc_add(p_xor, list(lhs, rhs), infer_fn = infer_binary_boolean)[[1L]]
}

p_shift_left <- Primitive("shift_left")
nvl_shift_left <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) {
    stablehlo::infer_types_shift_left(lhs, rhs)@items
  }
  graph_desc_add(p_shift_left, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_shift_right_logical <- Primitive("shift_right_logical")
nvl_shift_right_logical <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) {
    stablehlo::infer_types_shift_right_logical(lhs, rhs)@items
  }
  graph_desc_add(p_shift_right_logical, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_shift_right_arithmetic <- Primitive("shift_right_arithmetic")
nvl_shift_right_arithmetic <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) {
    stablehlo::infer_types_shift_right_arithmetic(lhs, rhs)@items
  }
  graph_desc_add(p_shift_right_arithmetic, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_atan2 <- Primitive("atan2")
nvl_atan2 <- function(lhs, rhs) {
  graph_desc_add(p_atan2, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
}

# unary math primitives ---------------------------------------------------------

p_abs <- Primitive("abs")
nvl_abs <- function(operand) {
  graph_desc_add(p_abs, list(operand), infer_fn = infer_unary)[[1L]]
}

p_sqrt <- Primitive("sqrt")
nvl_sqrt <- function(operand) {
  graph_desc_add(p_sqrt, list(operand), infer_fn = infer_unary)[[1L]]
}

p_rsqrt <- Primitive("rsqrt")
nvl_rsqrt <- function(operand) {
  graph_desc_add(p_rsqrt, list(operand), infer_fn = infer_unary)[[1L]]
}

p_log <- Primitive("log")
nvl_log <- function(operand) {
  graph_desc_add(p_log, list(operand), infer_fn = infer_unary)[[1L]]
}

p_tanh <- Primitive("tanh")
nvl_tanh <- function(operand) {
  graph_desc_add(p_tanh, list(operand), infer_fn = infer_unary)[[1L]]
}

p_tan <- Primitive("tan")
nvl_tan <- function(operand) {
  graph_desc_add(p_tan, list(operand), infer_fn = infer_unary)[[1L]]
}

p_floor <- Primitive("floor")
nvl_floor <- function(operand) {
  graph_desc_add(p_floor, list(operand), infer_fn = infer_unary)[[1L]]
}

p_ceil <- Primitive("ceil")
nvl_ceil <- function(operand) {
  graph_desc_add(p_ceil, list(operand), infer_fn = infer_unary)[[1L]]
}

p_sign <- Primitive("sign")
nvl_sign <- function(operand) {
  graph_desc_add(p_sign, list(operand), infer_fn = infer_unary)[[1L]]
}

p_exp <- Primitive("exp")
nvl_exp <- function(operand) {
  graph_desc_add(p_exp, list(operand), infer_fn = infer_unary)[[1L]]
}

p_round <- Primitive("round")
nvl_round <- function(operand, method = "nearest_even") {
  if (!(method %in% c("nearest_even", "afz"))) {
    cli_abort("method must be one of: 'nearest_even', 'afz', but is {method}")
  }
  infer_fn <- function(operand, method) {
    infer_unary(operand)
  }
  graph_desc_add(p_round, list(operand), list(method = method), infer_fn = infer_fn)[[1L]]
}

# dtype conversion ----------------------------------------------------------------

p_convert <- Primitive("convert")
nvl_convert <- function(operand, dtype) {
  infer_fn <- function(operand, dtype) {
    stablehlo::infer_types_convert(operand, dtype)@items
  }
  graph_desc_add(
    p_convert,
    list(operand),
    params = list(dtype = dtype),
    infer_fn = infer_fn
  )[[1L]]
}


p_select <- Primitive("select")
nvl_select <- function(pred, true_value, false_value) {
  infer_fn <- function(pred, true_value, false_value) {
    stablehlo::infer_types_select(pred, on_true = true_value, on_false = false_value)@items
  }
  graph_desc_add(p_select, list(pred, true_value, false_value), infer_fn = infer_fn)[[1L]]
}

# Higher order primitives -------------------------------------------------------

p_if <- HigherOrderPrimitive("if")
nvl_if <- function(pred, true, false) {
  true_expr <- rlang::enquo(true)
  false_expr <- rlang::enquo(false)

  # Build sub-graphs for each branch (no inputs, just capture closed-over values)
  # We need to ensure that constants that are captured in both branches receive the same
  # GraphValue if they capture the same constant

  current_desc <- .current_descriptor()
  desc_true <- local_descriptor()
  true_graph <- graphify(function() rlang::eval_tidy(true_expr), list(), desc = desc_true)
  desc_false <- local_descriptor()

  for (const in desc_true@constants) {
    get_box_or_register_const(desc_false, const)
  }
  false_graph <- graphify(function() rlang::eval_tidy(false_expr), list(), desc = desc_false)

  for (const in desc_false@constants) {
    get_box_or_register_const(current_desc, const)
  }

  if (!identical(true_graph@out_tree, false_graph@out_tree)) {
    cli_abort("true and false branches must have the same output structure")
  }

  infer_fn <- function(pred, true_graph, false_graph) {
    lapply(true_graph@outputs, \(out) st2vt(out@aval))
  }

  out <- graph_desc_add(
    p_if,
    list(pred),
    params = list(true_graph = true_graph, false_graph = false_graph),
    infer_fn = infer_fn
  )
  unflatten(true_graph@out_tree, out)
}

p_while <- HigherOrderPrimitive("while")
nvl_while <- function(init, cond, body) {
  if (!is.function(body)) {
    cli_abort("body must be a function")
  }
  if (!is.function(cond)) {
    cli_abort("cond must be a function")
  }

  state_names <- names(init)

  if (any(state_names == "")) {
    cli_abort("init must have only named arguments")
  }

  desc_cond <- local_descriptor()

  cond_graph <- graphify(cond, init, desc = desc_cond)

  desc_body <- local_descriptor()

  # ensure that constant ids are the same between cond and body
  # inputs don't matter, because we don't inline the sub-graphs into the parent graph
  for (const in desc_cond@constants) {
    get_box_or_register_const(desc_body, const)
  }
  body_graph <- graphify(body, init, desc_body)

  if (!identical(cond_graph@in_tree, body_graph@in_tree)) {
    cli_abort("cond and body must have the same input structure")
  }

  if (!identical(body_graph@in_tree, body_graph@out_tree)) {
    cli_abort("body must have the same input and output structure")
  }

  current_desc <- .current_descriptor()

  # now we register the constants of both sub-graphs (body includes cond's constants) into the graph
  for (const in body_graph@constants) {
    get_box_or_register_const(current_desc, const)
  }

  infer_fn <- function(..., cond_graph, body_graph) {
    outs <- list(...)
    outs_body <- lapply(body_graph@outputs, \(out) st2vt(out@aval))
    inputs_body <- lapply(body_graph@inputs, \(inp) st2vt(inp@aval))
    if (!identical(unname(outs), inputs_body) || !identical(inputs_body, outs_body)) {
      cli_abort("init must be the same as inputs and outputs of body")
    }
    return(outs)
  }

  out <- graph_desc_add(
    p_while,
    args = flatten(init),
    params = list(cond_graph = cond_graph, body_graph = body_graph),
    infer_fn = infer_fn
  )

  unflatten(body_graph@out_tree, out)
}
