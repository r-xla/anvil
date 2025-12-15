#' @include utils.R
#' @include type-converters.R

# Type inference helper functions for graph building
#' @importFrom stablehlo infer_types_generic_biv infer_types_generic_uni
#' @importFrom stablehlo infer_types_integerish_biv infer_types_boolean_uni
#' @importFrom stablehlo infer_types_compare infer_types_transpose infer_types_reshape
#' @importFrom stablehlo infer_types_broadcast_in_dim infer_types_convert
#' @importFrom stablehlo infer_types_dot_general infer_types_select

infer_binary <- function(lhs, rhs) {
  both_ambiguous <- lhs@ambiguous && rhs@ambiguous
  out <- stablehlo::infer_types_generic_biv(st2vt(lhs), st2vt(rhs))@items[[1L]]
  out <- vt2st(out)
  out@ambiguous <- both_ambiguous
  list(out)
}

# boolean is i1 -> integerish
infer_binary_integerish <- function(lhs, rhs) {
  both_ambiguous <- lhs@ambiguous && rhs@ambiguous
  out <- stablehlo::infer_types_integerish_biv(st2vt(lhs), st2vt(rhs))@items[[1L]]
  out <- vt2st(out)
  out@ambiguous <- both_ambiguous
  list(out)
}

infer_unary <- function(operand) {
  out <- stablehlo::infer_types_generic_uni(st2vt(operand))@items[[1L]]
  out <- vt2st(out)
  out@ambiguous <- operand@ambiguous
  list(out)
}

make_binary_op <- function(prim) {
  function(lhs, rhs) {
    graph_desc_add(prim, list(lhs, rhs), infer_fn = infer_binary, infer_hlo = FALSE)[[1L]]
  }
}

make_binary_integerish_op <- function(prim) {
  function(lhs, rhs) {
    graph_desc_add(prim, list(lhs, rhs), infer_fn = infer_binary_integerish, infer_hlo = FALSE)[[1L]]
  }
}

infer_unary_integerish <- function(operand) {
  out <- stablehlo::infer_types_integerish_uni(st2vt(operand))@items[[1L]]
  out <- vt2st(out)
  out@ambiguous <- operand@ambiguous
  list(out)
}

make_unary_op <- function(prim) {
  function(operand) {
    graph_desc_add(prim, list(operand), infer_fn = infer_unary, infer_hlo = FALSE)[[1L]]
  }
}

make_unary_integerish_op <- function(prim) {
  function(operand) {
    graph_desc_add(prim, list(operand), infer_fn = infer_unary_integerish, infer_hlo = FALSE)[[1L]]
  }
}

infer_binary_integerish <- function(lhs, rhs) {
  stablehlo::infer_types_integerish_biv(lhs, rhs)@items
}

infer_unary_integerish <- function(operand) {
  stablehlo::infer_types_integerish_uni(operand)@items
}

infer_reduce <- function(operand, dims, drop) {
  old_shape <- shape(operand)
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(ShapedTensor(
    dtype = dtype(operand),
    shape = Shape(new_shape),
    ambiguous = operand@ambiguous
  ))
}

infer_reduce_boolean <- function(operand, dims, drop) {
  old_shape <- shape(operand)
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(ShapedTensor(
    dtype = "pred",
    shape = Shape(new_shape),
    ambiguous = operand@ambiguous
  ))
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

p_full <- Primitive("full")
nvl_full <- function(value, shape, dtype) {
  infer_full <- function(value, shape, dtype) {
    list(ShapedTensor(dtype = as_dtype(dtype), shape = shape, ambiguous = FALSE))
  }
  graph_desc_add(
    p_full,
    list(),
    params = list(value = value, dtype = dtype, shape = shape),
    infer_fn = infer_full,
    infer_hlo = FALSE
  )[[1L]]
}

p_add <- Primitive("add")
nvl_add <- make_binary_op(p_add)

p_mul <- Primitive("mul")
nvl_mul <- make_binary_op(p_mul)

p_sub <- Primitive("sub")
nvl_sub <- make_binary_op(p_sub)

p_neg <- Primitive("negate")
nvl_neg <- make_unary_op(p_neg)

p_div <- Primitive("divide")
nvl_div <- make_binary_op(p_div)

p_pow <- Primitive("power")
nvl_pow <- make_binary_op(p_pow)

p_broadcast_in_dim <- Primitive("broadcast_in_dim")

#' @importFrom stablehlo r_to_constant
nvl_broadcast_in_dim <- function(operand, shape_out, broadcast_dimensions) {
  infer_fn <- function(operand, shape_out, broadcast_dimensions) {
    bd_attr <- r_to_constant(
      as.integer(broadcast_dimensions - 1L),
      dtype = "i64",
      shape = length(broadcast_dimensions)
    )
    out <- stablehlo::infer_types_broadcast_in_dim(
      st2vt(operand),
      broadcast_dimensions = bd_attr,
      shape_out = shape_out
    )@items[[1L]]
    out <- vt2st(out)
    out@ambiguous <- operand@ambiguous
    list(out)
  }
  graph_desc_add(
    p_broadcast_in_dim,
    list(operand),
    params = list(
      shape_out = shape_out,
      broadcast_dimensions = broadcast_dimensions
    ),
    infer_fn = infer_fn,
    infer_hlo = FALSE
  )[[1L]]
}

p_dot_general <- Primitive("dot_general")
nvl_dot_general <- function(lhs, rhs, contracting_dims, batching_dims) {
  infer_fn <- function(lhs, rhs, contracting_dims, batching_dims) {
    ddn <- stablehlo::DotDimensionNumbers(
      contracting_dims = lapply(contracting_dims, \(x) x - 1L),
      batching_dims = lapply(batching_dims, \(x) x - 1L)
    )
    out <- stablehlo::infer_types_dot_general(st2vt(lhs), st2vt(rhs), dot_dimension_numbers = ddn)@items[[1L]]
    list(vt2st(out))
  }
  graph_desc_add(
    p_dot_general,
    list(lhs, rhs),
    list(contracting_dims = contracting_dims, batching_dims = batching_dims),
    infer_fn = infer_fn,
    infer_hlo = FALSE
  )[[1L]]
}

p_transpose <- Primitive("transpose")
nvl_transpose <- function(operand, permutation) {
  infer_fn <- function(operand, permutation) {
    perm_attr <- r_to_constant(
      as.integer(permutation - 1L),
      dtype = "i64",
      shape = length(permutation)
    )
    out <- stablehlo::infer_types_transpose(st2vt(operand), permutation = perm_attr)@items[[1L]]
    out <- vt2st(out)
    out@ambiguous <- operand@ambiguous
    list(out)
  }
  graph_desc_add(
    p_transpose,
    list(operand),
    list(permutation = permutation),
    infer_fn = infer_fn,
    infer_hlo = FALSE
  )[[1L]]
}

p_reshape <- Primitive("reshape")
nvl_reshape <- function(operand, shape) {
  infer_fn <- function(operand, shape) {
    out <- stablehlo::infer_types_reshape(st2vt(operand), shape_out = shape)@items[[1L]]
    out <- vt2st(out)
    out@ambiguous <- operand@ambiguous
    list(out)
  }
  graph_desc_add(
    p_reshape,
    list(operand),
    params = list(shape = shape),
    infer_fn = infer_fn,
    infer_hlo = FALSE
  )[[1L]]
}

p_concatenate <- Primitive("concatenate")
nvl_concatenate <- function(..., dimension) {
  dots <- list(...)
  infer_fn <- function(..., dimension) {
    stablehlo::infer_types_concatenate(..., dimension = dimension)@items
  }
  graph_desc_add(
    p_concatenate,
    args = dots,
    params = list(dimension = dimension),
    infer_fn = infer_fn
  )[[1L]]
}

p_slice <- Primitive("slice")
nvl_slice <- function(operand, start_indices, limit_indices, strides) {
  infer_fn <- function(operand, start_indices, limit_indices, strides) {
    start_indices <- r_to_constant(start_indices - 1L, dtype = "i64", shape = length(start_indices))
    limit_indices <- r_to_constant(limit_indices, dtype = "i64", shape = length(limit_indices))
    strides <- r_to_constant(strides, dtype = "i64", shape = length(strides))
    stablehlo::infer_types_slice(operand, start_indices, limit_indices, strides)@items
  }
  graph_desc_add(
    p_slice,
    args = list(
      operand = operand
    ),
    params = list(
      start_indices = start_indices,
      limit_indices = limit_indices,
      strides = strides
    ),
    infer_fn = infer_fn
  )[[1L]]
}

# reduction operators

make_reduce_op <- function(prim, infer_fn = infer_reduce) {
  function(operand, dims, drop = TRUE) {
    graph_desc_add(
      prim,
      list(operand),
      params = list(dims = dims, drop = drop),
      infer_fn = infer_fn,
      infer_hlo = FALSE
    )[[1L]]
  }
}

p_reduce_sum <- Primitive("sum")
nvl_reduce_sum <- make_reduce_op(p_reduce_sum)

p_reduce_prod <- Primitive("prod")
nvl_reduce_prod <- make_reduce_op(p_reduce_prod)

p_reduce_max <- Primitive("max")
nvl_reduce_max <- make_reduce_op(p_reduce_max)

p_reduce_min <- Primitive("min")
nvl_reduce_min <- make_reduce_op(p_reduce_min)

p_reduce_any <- Primitive("any")
nvl_reduce_any <- make_reduce_op(p_reduce_any, infer_reduce_boolean)

p_reduce_all <- Primitive("all")
nvl_reduce_all <- make_reduce_op(p_reduce_all, infer_reduce_boolean)

# comparison primitives --------------------------------------------------------

infer_compare <- function(lhs, rhs, comparison_direction) {
  out <- stablehlo::infer_types_compare(st2vt(lhs), st2vt(rhs), comparison_direction, "FLOAT")@items[[1L]]
  list(vt2st(out))
}

make_compare_op <- function(prim, direction) {
  infer_fn <- function(lhs, rhs) infer_compare(lhs, rhs, direction)
  function(lhs, rhs) {
    graph_desc_add(prim, list(lhs, rhs), infer_fn = infer_fn, infer_hlo = FALSE)[[1L]]
  }
}

p_eq <- Primitive("equal")
nvl_eq <- make_compare_op(p_eq, "EQ")

p_ne <- Primitive("not_equal")
nvl_ne <- make_compare_op(p_ne, "NE")

p_gt <- Primitive("greater")
nvl_gt <- make_compare_op(p_gt, "GT")

p_ge <- Primitive("greater_equal")
nvl_ge <- make_compare_op(p_ge, "GE")

p_lt <- Primitive("less")
nvl_lt <- make_compare_op(p_lt, "LT")

p_le <- Primitive("less_equal")
nvl_le <- make_compare_op(p_le, "LE")

# additional simple binary primitives -----------------------------------------

p_max <- Primitive("maximum")
nvl_max <- make_binary_op(p_max)

p_min <- Primitive("minimum")
nvl_min <- make_binary_op(p_min)

p_remainder <- Primitive("remainder")
nvl_remainder <- make_binary_op(p_remainder)

p_and <- Primitive("and")

nvl_and <- make_binary_boolean_op(p_and)

p_not <- Primitive("not")
nvl_not <- make_unary_boolean_op(p_not)

p_or <- Primitive("or")
nvl_or <- make_binary_boolean_op(p_or)

p_xor <- Primitive("xor")
nvl_xor <- make_binary_boolean_op(p_xor)

infer_shift <- function(lhs, rhs, shift_fn) {
  both_ambiguous <- lhs@ambiguous && rhs@ambiguous
  out <- shift_fn(st2vt(lhs), st2vt(rhs))@items[[1L]]
  out <- vt2st(out)
  out@ambiguous <- both_ambiguous
  list(out)
}

p_shift_left <- Primitive("shift_left")
nvl_shift_left <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_left)
  graph_desc_add(p_shift_left, list(lhs, rhs), infer_fn = infer_fn, infer_hlo = FALSE)[[1L]]
}

p_shift_right_logical <- Primitive("shift_right_logical")
nvl_shift_right_logical <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_logical)
  graph_desc_add(p_shift_right_logical, list(lhs, rhs), infer_fn = infer_fn, infer_hlo = FALSE)[[1L]]
}

p_shift_right_arithmetic <- Primitive("shift_right_arithmetic")
nvl_shift_right_arithmetic <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_arithmetic)
  graph_desc_add(p_shift_right_arithmetic, list(lhs, rhs), infer_fn = infer_fn, infer_hlo = FALSE)[[1L]]
}

p_atan2 <- Primitive("atan2")
nvl_atan2 <- make_binary_op(p_atan2)

p_bitcast_convert <- Primitive("bitcast_convert")
nvl_bitcast_convert <- function(operand, dtype) {
  infer_fn <- function(operand, dtype) {
    stablehlo::infer_types_bitcast_convert(operand, dtype)@items
  }
  graph_desc_add(p_bitcast_convert, list(operand), params = list(dtype = dtype), infer_fn = infer_fn)[[1L]]
}

# unary math primitives ---------------------------------------------------------

p_abs <- Primitive("abs")
nvl_abs <- make_unary_op(p_abs)

p_sqrt <- Primitive("sqrt")
nvl_sqrt <- make_unary_op(p_sqrt)

p_rsqrt <- Primitive("rsqrt")
nvl_rsqrt <- make_unary_op(p_rsqrt)

p_log <- Primitive("log")
nvl_log <- make_unary_op(p_log)

p_tanh <- Primitive("tanh")
nvl_tanh <- make_unary_op(p_tanh)

p_tan <- Primitive("tan")
nvl_tan <- make_unary_op(p_tan)

p_tan <- Primitive("tan")
nvl_tan <- function(operand) {
  graph_desc_add(p_tan, list(operand), infer_fn = infer_unary)[[1L]]
}

p_sine <- Primitive("sine")
nvl_sine <- function(operand) {
  graph_desc_add(p_sine, list(operand), infer_fn = infer_unary)[[1L]]
}

p_cosine <- Primitive("cosine")
nvl_cosine <- function(operand) {
  graph_desc_add(p_cosine, list(operand), infer_fn = infer_unary)[[1L]]
}

p_floor <- Primitive("floor")
nvl_floor <- make_unary_op(p_floor)

p_ceil <- Primitive("ceil")
nvl_ceil <- make_unary_op(p_ceil)

p_sign <- Primitive("sign")
nvl_sign <- make_unary_op(p_sign)

p_exp <- Primitive("exp")
nvl_exp <- make_unary_op(p_exp)

p_round <- Primitive("round")
nvl_round <- function(operand, method = "nearest_even") {
  if (!(method %in% c("nearest_even", "afz"))) {
    cli_abort("method must be one of: 'nearest_even', 'afz', but is {method}")
  }
  infer_fn <- function(operand, method) {
    infer_unary(operand)
  }
  graph_desc_add(p_round, list(operand), list(method = method), infer_fn = infer_fn, infer_hlo = FALSE)[[1L]]
}

# dtype conversion ----------------------------------------------------------------

p_convert <- Primitive("convert")
nvl_convert <- function(operand, dtype, ambiguous = FALSE) {
  dtype <- as_dtype(dtype)
  infer_fn <- function(operand, dtype, ambiguous) {
    list(ShapedTensor(
      dtype = dtype,
      shape = Shape(shape(operand)),
      ambiguous = ambiguous
    ))
  }
  graph_desc_add(
    p_convert,
    list(operand),
    params = list(dtype = dtype, ambiguous = ambiguous),
    infer_fn = infer_fn,
    infer_hlo = FALSE
  )[[1L]]
}


p_select <- Primitive("select")
nvl_select <- function(pred, true_value, false_value) {
  infer_fn <- function(pred, true_value, false_value) {
    both_ambiguous <- true_value@ambiguous && false_value@ambiguous
    out <- stablehlo::infer_types_select(
      st2vt(pred),
      on_true = st2vt(true_value),
      on_false = st2vt(false_value)
    )@items[[1L]]
    out <- vt2st(out)
    out@ambiguous <- both_ambiguous
    list(out)
  }
  graph_desc_add(p_select, list(pred, true_value, false_value), infer_fn = infer_fn, infer_hlo = FALSE)[[1L]]
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
  true_graph <- trace_fn(function() rlang::eval_tidy(true_expr), list(), desc = desc_true)
  desc_false <- local_descriptor()

  # TODO: Apply promotion rules to the outputs of the branches

  for (const in desc_true@constants) {
    get_box_or_register_const(desc_false, const)
  }
  false_graph <- trace_fn(function() rlang::eval_tidy(false_expr), list(), desc = desc_false)

  for (const in desc_false@constants) {
    get_box_or_register_const(current_desc, const)
  }

  if (!identical(true_graph@out_tree, false_graph@out_tree)) {
    cli_abort("true and false branches must have the same output structure")
  }

  infer_fn <- function(pred, true_graph, false_graph) {
    # the returned values might have different ambiguity, so we need to handle it
    # an output is ambiguous if it's type is ambiguous in both branches
    lapply(seq_along(true_graph@outputs), function(i) {
      aval_true <- true_graph@outputs[[i]]@aval
      aval_false <- true_graph@outputs[[i]]@aval
      if (aval_true@ambiguous && aval_false@ambiguous) {
        return(aval_true)
      }

      aval_true@ambiguous <- FALSE
      return(aval_true)
    })
  }

  out <- graph_desc_add(
    p_if,
    list(pred),
    params = list(true_graph = true_graph, false_graph = false_graph),
    infer_fn = infer_fn,
    infer_hlo = FALSE,
    desc = current_desc
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

  current_desc <- .current_descriptor()

  desc_cond <- local_descriptor()

  cond_graph <- trace_fn(cond, init, desc = desc_cond)

  desc_body <- local_descriptor()

  # ensure that constant ids are the same between cond and body
  # inputs don't matter, because we don't inline the sub-graphs into the parent graph
  for (const in desc_cond@constants) {
    get_box_or_register_const(desc_body, const)
  }
  body_graph <- trace_fn(body, init, desc_body)

  if (!identical(cond_graph@in_tree, body_graph@in_tree)) {
    cli_abort("cond and body must have the same input structure")
  }

  if (!identical(body_graph@in_tree, body_graph@out_tree)) {
    cli_abort("body must have the same input and output structure")
  }

  # now we register the constants of both sub-graphs (body includes cond's constants) into the graph
  for (const in body_graph@constants) {
    get_box_or_register_const(current_desc, const)
  }

  infer_fn <- function(..., cond_graph, body_graph) {
    outs <- list(...)
    outs_body <- lapply(body_graph@outputs, \(out) out@aval)
    inputs_body <- lapply(body_graph@inputs, \(inp) inp@aval)
    # == ignores ambiguity
    if (!all(sapply(seq_along(outs), \(i) outs[[i]] == outs_body[[i]]))) {
      cli_abort("outs must be have same type as outs_body")
    }
    if (!all(sapply(seq_along(inputs_body), \(i) inputs_body[[i]] == outs_body[[i]]))) {
      cli_abort("inputs_body must be have same type as outs_body")
    }
    # function might change the ambiguity, so we return the body outputs and not the inputs
    return(outs_body)
  }

  out <- graph_desc_add(
    p_while,
    args = flatten(init),
    params = list(cond_graph = cond_graph, body_graph = body_graph),
    infer_fn = infer_fn,
    infer_hlo = FALSE,
    desc = current_desc
  )

  unflatten(body_graph@out_tree, out)
}

# RNG primitives
p_rng_bit_generator <- Primitive("rng_bit_generator")
nvl_rng_bit_generator <- function(initial_state, rng_algorithm = "THREE_FRY", dtype, shape_out) {
  infer_fn <- function(initial_state, rng_algorithm, dtype, shape_out) {
    stablehlo::infer_types_rng_bit_generator(initial_state, rng_algorithm, dtype, shape_out)@items
  }
  graph_desc_add(
    p_rng_bit_generator,
    list(initial_state),
    params = list(rng_algorithm = rng_algorithm, dtype = dtype, shape_out = shape_out),
    infer_fn = infer_fn
  )
}
