#' @include repr.R
#' @importFrom S7 new_class new_property method
#' @include list_of.R primitives.R
#' @include ops.R
NULL

Variable <- new_class(
  "Variable",
  properties = list(
    aval = ShapedArray,
    # to get reference semantics and we can uniquely identify the variable
    .id = class_environment
  ),
  constructor = function(aval) {
    S7::new_object(S7::S7_object(), aval = aval, .id = new.env(size = 0L))
  }
)

Literal <- new_class(
  "Literal",
  properties = list(
    value = Array,
    aval = ShapedArray
  ),
  constructor = function(value) {
    if (!inherits(value, "nvl_array")) {
      stop("Literals can currently only be nvl_arrays")
    }
    aval <- ShapedArray(
      dtype_from_buffer(value),
      Shape(shape(value))
    )
    S7::new_object(
      S7::object(),
      value = value,
      aval = aval
    )
  }
)

Atom <- new_union(Variable, Literal)
Atoms <- new_list_of("Atoms", Atom)

Variables <- new_list_of("Variables", Variable)
Params <- new_list_of("Params", S7::class_any)

Equation <- new_class(
  "Equation",
  properties = list(
    primitive = AnvilOp, # This is an S7 generic
    inputs = Atoms,
    params = Params,
    out_binders = Variables
  )
)

Equations <- new_list_of("Equations", Equation)

Expr <- new_class(
  "Expr",
  properties = list(
    in_binders = Variables,
    equations = Equations,
    outputs = Variables,
    id = S7::new_property(S7::class_environment, default = new.env(size = 0L))
  )
)

method(`==`, list(Expr, Expr)) <- function(e1, e2) {
  identical(e1@id, e2@id)
}

method(hash, Expr) <- function(x) {
  hash(x@id)
}

ShapedArrays <- new_list_of("ShapedArrays", ShapedArray)

ExprType <- new_class(
  "ExprType",
  properties = list(
    in_types = ShapedArrays,
    out_types = ShapedArrays
  )
)

method(repr, ExprType) <- function(x) {
  vrepr <- function(x) {
    paste(sapply(x, repr), collapse = ", ")
  }
  sprintf("(%s) -> (%s)", vrepr(x@in_types), vrepr(x@out_types))
}

typecheck_expr <- function(expr) {
  env <- set()

  for (v in expr@in_binders) {
    if (set_has(env, v)) {
      stop("Duplicate variable")
    }
    set_add(env, v)
  }
  for (eqn in expr@equations) {
    in_types <- lapply(eqn@inputs, typecheck_atom, env = env)
    out_types <- do.call(
      type_inference_rules[[eqn@primitive@name]],
      c(in_types, eqn@params)
    )
    lapply(seq_along(eqn@out_binders), function(i) {
      if (eqn@out_binders[[i]]@aval != out_types[[i]]) {
        stop("Type mismatch")
      }
    })
    for (out_binder in eqn@out_binders) {
      if (set_has(env, out_binder)) {
        stop("Duplicate variable")
      }
      set_add(env, out_binder)
    }
  }
  in_types <- lapply(expr@in_binders, typecheck_atom, env = env)
  out_types <- lapply(expr@outputs, typecheck_atom, env = env)
  ExprType(in_types, out_types)
}

typecheck_atom <- function(x, env) {
  if (inherits(x, Variable)) {
    if (!set_has(env, x)) {
      stop("Unbound variable")
    }
    x@aval
  } else if (inherits(x, Literal)) {
    raise_to_shaped(get_aval(x@value))
  } else {
    stop("Unknown type")
  }
}

eval_expr <- function(expr, args) {
  env <- hashtab()
  read <- function(x) {
    if (inherits(x, Variable)) {
      env[[x]]
    } else {
      x@value
    }
  }
  write <- function(var, value) {
    if (!is.null(env[[var]])) {
      stop("Duplicate variable")
    }
    env[[var]] <- value
  }
  if (length(expr@in_binders) != length(args)) {
    stop("Wrong number of arguments")
  }
  for (i in seq_along(expr@in_binders)) {
    write(expr@in_binders[[i]], args[[i]])
  }
  for (eqn in expr@equations) {
    in_vals <- lapply(eqn@inputs, read)
    out_vals <- do.call(eqn@primitive@name, c(in_vals, eqn@params))

    for (i in seq_along(eqn@out_binders)) {
      write(eqn@out_binders[[i]], out_vals[[i]])
    }
  }
  lapply(expr@outputs, read)
}

expr_to_function <- function(expr) {
  function(...) {
    eval_expr(expr, list(...))
  }
}
