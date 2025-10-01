#' @include ir.R
NULL

method(repr, IRVariable) <- function(x) {
  paste0("%", stablehlo::repr_env2name(x@id))
}

method(print, IRVariable) <- function(x, ...) {
  cat(repr(x), "\n")
}

method(repr, IRAtoms) <- function(x) {
  paste0(sapply(x@items, repr), collapse = ", ")
}

method(repr, IRVariables) <- function(x) {
  paste0(sapply(x@items, repr), collapse = ", ")
}

method(repr, IRLiteral) <- function(x) {
  as.character(as_array(x@value))
}

method(print, IRLiteral) <- function(x, ...) {
  cat(repr(x), "\n")
}

method(print, IRAtoms) <- function(x, ...) {
  cat(repr(x), "\n")
}

method(print, IRVariables) <- function(x, ...) {
  cat(repr(x), "\n")
}

method(repr, IRParams) <- function(x) {
  # TODO: print params
  if (length(x) > 1L) {
    sprintf("... (%s params)", length(x))
  } else if (length(x) == 1L) {
    "... (1 param)"
  }
}

method(print, IRParams) <- function(x, ...) {
  cat(repr(x), "\n")
}

method(repr, IREquation) <- function(x) {
  lhs <- repr(x@out_binders)
  inputs <- repr(x@inputs)
  params <- repr(x@params)
  params <- if (!is.null(params)) {
    paste0(", ", params)
  }
  paste0(lhs, " = ", x@primitive@name, " ", inputs, params)
}

method(print, IREquation) <- function(x, ...) {
  cat(repr(x), "\n")
}

method(repr, IREquations) <- function(x) {
  paste0(vapply(x@items, repr, character(1)), collapse = "\n")
}

method(print, IREquations) <- function(x, ...) {
  cat(repr(x), "\n")
}

method(repr, IRExpr) <- function(x) {
  in_binders <- repr(x@in_binders)
  body <- if (length(x@equations@items)) {
    paste0(
      "  let\n    ",
      paste0(
        vapply(x@equations@items, repr, character(1)),
        collapse = "\n    "
      ),
      "\n  "
    )
  } else {
    "  "
  }
  outs <- repr(x@outputs)
  paste0(
    "{ lambda ",
    in_binders,
    " .\n",
    body,
    "in ( ",
    outs,
    " ) }"
  )
}

method(print, IRExpr) <- function(x, ...) {
  cat(repr(x), "\n")
}

method(repr, IRType) <- function(x) {
  vrepr <- function(x) {
    paste(sapply(x, repr), collapse = ", ")
  }
  sprintf("(%s) -> (%s)", vrepr(x@in_types), vrepr(x@out_types))
}

method(print, IRType) <- function(x, ...) {
  cat(repr(x), "\n")
}
