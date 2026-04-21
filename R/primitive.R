#' @title AnvilPrimitive
#' @description
#' Primitive interpretation rule.
#' Note that `[[` and `[[<-` access the interpretation rules.
#' To access other fields, use `$` and `$<-`.
#'
#' A primitive is considered higher-order if it contains subgraphs.
#' @param name (`character()`)\cr
#'   The name of the primitive.
#' @param subgraphs (`character()`)\cr
#'   Names of parameters that are subgraphs. Only used if `higher_order = TRUE`.
#' @return (`AnvilPrimitive`)
#' @export
AnvilPrimitive <- function(name, subgraphs = character()) {
  checkmate::assert_string(name)
  checkmate::assert_character(subgraphs)

  env <- new.env(parent = emptyenv())
  env$name <- name
  env$rules <- list()
  env$subgraphs <- subgraphs

  structure(env, class = "AnvilPrimitive")
}


primitive_env <- new.env(parent = emptyenv())

is_higher_order_primitive <- function(x) {
  if (inherits(x, "JitPrimitive")) x <- attr(x, "primitive")
  length(x$subgraphs) > 0L
}


#' @method [[<- AnvilPrimitive
#' @export
`[[<-.AnvilPrimitive` <- function(x, name, value) {
  if (name %in% globals$interpretation_rules) {
    x$rules[[name]] <- value
  } else {
    cli_abort("Invalid field name {.field {name}} for primitive {.field {x$name}}")
  }
  x
}

#' @method [[ AnvilPrimitive
#' @export
`[[.AnvilPrimitive` <- function(x, name) {
  if (name %in% globals$interpretation_rules) {
    return(x$rules[[name]])
  }
  cli_abort("Invalid field name {.field {name}} for primitive {.field {x$name}}")
}

#' @method print AnvilPrimitive
#' @export
print.AnvilPrimitive <- function(x, ...) {
  cat(sprintf("<AnvilPrimitive:%s>\n", x$name))
  invisible(x)
}

#' @method [[ JitPrimitive
#' @export
`[[.JitPrimitive` <- function(x, name) {
  attr(x, "primitive")[[name]]
}

#' @method [[<- JitPrimitive
#' @export
`[[<-.JitPrimitive` <- function(x, name, value) {
  attr(x, "primitive")[[name]] <- value
  x
}

#' @method print JitPrimitive
#' @export
print.JitPrimitive <- function(x, ...) {
  print(attr(x, "primitive"))
  invisible(x)
}

#' @title Create a Primitive
#' @description
#' Builds an [`AnvilPrimitive`] metadata object, wraps `fn` with [`jit()`]
#' (backend `"auto"`), attaches the metadata via `attr(., "primitive")`,
#' prepends class `"JitPrimitive"`, and (by default) registers the result
#' under `name` in the primitive registry.
#' @param name (`character(1)`)\cr
#'   Primitive name.
#' @param fn (`function`)\cr
#'   Body of the primitive. Its formals become the formals of the returned
#'   JIT-compiled callable. Inside `fn`, identify the primitive by passing
#'   the name string to [`graph_desc_add()`] (not the `fn` itself).
#' @param subgraphs (`character()`)\cr
#'   Names of parameters that are subgraphs (for higher-order primitives).
#' @param static (`character()` | `integer()`)\cr
#'   Passed to [`jit()`].
#' @param device (`NULL` | `character(1)` | `device_arg()`)\cr
#'   Passed to [`jit()`]. Useful for primitives with no array inputs
#'   (e.g. `prim_fill`) where the device must come from an explicit argument.
#' @param register (`logical(1)`)\cr
#'   If `TRUE` (default), register the result under `name` in the primitive
#'   registry.
#' @return A callable of class `c("JitPrimitive", "JitFunction")`.
#' @export
new_primitive <- function(name, fn, subgraphs = character(),
                          static = character(), device = NULL,
                          register = TRUE) {
  checkmate::assert_string(name)
  checkmate::assert_function(fn)
  checkmate::assert_character(subgraphs)
  checkmate::assert_flag(register)

  primitive <- AnvilPrimitive(name, subgraphs = subgraphs)
  jit_fn <- jit(fn, static = static, backend = "auto", device = device)
  attr(jit_fn, "primitive") <- primitive
  class(jit_fn) <- c("JitPrimitive", class(jit_fn))

  if (register) {
    assign(name, jit_fn, envir = primitive_env)
  }

  jit_fn
}

#' @title Get Subgraphs from Higher-Order Primitive
#' @description
#' Extracts subgraphs from the parameters of a higher-order primitive call.
#' @param call (`PrimitiveCall`)\cr
#'   The primitive call.
#' @return (`list(AnvilGraph)`)\cr
#'   List of subgraphs found in the parameters.
#' @export
subgraphs <- function(call) {
  p <- call$primitive
  if (inherits(p, "JitPrimitive")) p <- attr(p, "primitive")
  if (!is_higher_order_primitive(p)) return(list())

  stats::setNames(
    lapply(p$subgraphs, \(sg) call$params[[sg]]),
    p$subgraphs
  )
}
