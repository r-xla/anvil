#' @include repr.R
#' @importFrom S7 new_class new_property method
#' @include list_of.R primitives.R
#' @include ops.R
#' @include tensor.R
NULL

IRVariable <- new_class(
  "IRVariable",
  properties = list(
    aval = ShapedTensor,
    # to get reference semantics and we can uniquely identify the variable
    id = S7::new_property(S7::class_environment)
  ),
  constructor = function(aval) {
    S7::new_object(S7::S7_object(), aval = aval, id = zero_env())
  }
)

IRLiteral <- new_class(
  "IRLiteral",
  properties = list(
    value = Tensor,
    aval = ShapedTensor
  ),
  constructor = function(value) {
    if (!inherits(value, "AnvilTensor")) {
      cli_abort("IRLiterals can currently only be nv_tensors")
    }
    if (tengen::ndims(value) != 0L) {
      cli_abort("Only scalars can be IRLiterals")
    }
    aval <- ShapedTensor(
      dtype_from_buffer(value),
      Shape(integer())
    )
    S7::new_object(
      S7::S7_object(),
      value = value,
      aval = aval
    )
  }
)

IRAtom <- new_union(IRVariable, IRLiteral)
IRAtoms <- new_list_of("IRAtoms", IRAtom)


IRVariables <- new_list_of("IRVariables", IRVariable)

IRParams <- new_list_of("IRParams", S7::class_any)


IREquation <- new_class(
  "IREquation",
  properties = list(
    primitive = Primitive,
    inputs = IRAtoms,
    params = IRParams,
    out_binders = IRVariables
  )
)

method(`==`, list(IREquation, IREquation)) <- function(e1, e2) {
  identical(e1@primitive, e2@primitive) &&
    identical(e1@inputs, e2@inputs) &&
    identical(e1@params, e2@params) &&
    identical(e1@out_binders, e2@out_binders)
}

IREquations <- new_list_of("IREquations", IREquation)

IRExpr <- new_class(
  "IRExpr",
  properties = list(
    in_binders = IRVariables,
    equations = IREquations,
    outputs = IRVariables,
    # TODO: why is this needed?
    id = class_environment
  ),
  constructor = function(in_binders, equations, outputs) {
    S7::new_object(
      S7::S7_object(),
      in_binders = in_binders,
      equations = equations,
      outputs = outputs,
      id = new.env(size = 0L)
    )
  }
)

method(`==`, list(IRExpr, IRExpr)) <- function(e1, e2) {
  identical(e1@id, e2@id)
}

method(hash, IRExpr) <- function(x) {
  hash(x@id)
}

ShapedTensors <- new_list_of("ShapedTensors", ShapedTensor)

IRType <- new_class(
  "IRType",
  properties = list(
    in_types = ShapedTensors,
    out_types = ShapedTensors
  )
)


#typecheck_ir <- function(ir) {
#  env <- set()
#
#  for (v in ir@in_binders) {
#    if (set_has(env, v)) {
#      cli_abort("Duplicate variable")
#    }
#    set_add(env, v)
#  }
#  for (eqn in ir@equations) {
#    in_types <- lapply(eqn@inputs, typecheck_atom, env = env)
#    out_types <- do.call(
#      type_inference_rules[[eqn@primitive@name]],
#      c(in_types, eqn@params)
#    )
#    lapply(seq_along(eqn@out_binders), function(i) {
#      if (eqn@out_binders[[i]]@aval != out_types[[i]]) {
#        cli_abort("Type mismatch")
#      }
#    })
#    for (out_binder in eqn@out_binders) {
#      if (set_has(env, out_binder)) {
#        cli_abort("Duplicate variable")
#      }
#      set_add(env, out_binder)
#    }
#  }
#  in_types <- lapply(ir@in_binders, typecheck_atom, env = env)
#  out_types <- lapply(ir@outputs, typecheck_atom, env = env)
#  IRType(in_types, out_types)
#}
#
#typecheck_atom <- function(x, env) {
#  if (inherits(x, IRVariable)) {
#    if (!set_has(env, x)) {
#      cli_abort("Unbound variable")
#    }
#    x@aval
#  } else if (inherits(x, IRLiteral)) {
#    raise_to_shaped(aval(x@value))
#  } else {
#    cli_abort("Unknown type")
#  }
#}

eval_ir <- function(ir, args) {
  env <- hashtab()
  read <- function(x) {
    if (inherits(x, IRVariable)) {
      env[[x]]
    } else {
      x@value
    }
  }
  write <- function(var, value) {
    if (!is.null(env[[var]])) {
      cli_abort("Duplicate variable")
    }
    env[[var]] <- value
  }
  if (length(ir@in_binders) != length(args)) {
    cli_abort("Wrong number of arguments")
  }
  for (i in seq_along(ir@in_binders)) {
    write(ir@in_binders[[i]], args[[i]])
  }
  for (eqn in ir@equations) {
    in_vals <- lapply(eqn@inputs, read)
    out_vals <- do.call(eqn@primitive@name, c(in_vals, eqn@params))

    for (i in seq_along(eqn@out_binders)) {
      write(eqn@out_binders[[i]], out_vals[[i]])
    }
  }
  lapply(ir@outputs, read)
}

# TODO: Can probably make this into a proper R functioin that does not need
# eval_ir
ir_to_function <- function(ir) {
  function(...) {
    eval_ir(ir, list(...))
  }
}


#' Get a unique ID
#'
#' This is used in the IR to uniquely identify:
#' * Boxes
#' * Constant Values
#'
#' @noRd
ir_id <- new_generic("ir_id", "x", function(x) {
  S7::S7_dispatch()
})

method(ir_id, class_environment) <- function(x) {
  addr <- rlang::obj_address(x)
  # Remove the "0x" prefix before converting to integer
  addr_clean <- sub("^0x", "", addr)
  strtoi(addr_clean, base = 16L)
}

method(ir_id, Tensor) <- function(x) {
  rlang::obj_address(x)
}
