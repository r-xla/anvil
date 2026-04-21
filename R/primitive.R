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
  env$higher_order <- length(subgraphs) > 0L
  if (env$higher_order) {
    env$subgraphs <- subgraphs
  }

  structure(env, class = "AnvilPrimitive")
}


prim_dict <- new.env(parent = emptyenv())

#' @title Register a Primitive
#' @description
#' Register a primitive.
#' @param name (`character()`)\cr
#'   The name of the primitive.
#' @param primitive (`AnvilPrimitive`)\cr
#'   The primitive to register.
#' @param overwrite (`logical(1)`)\cr
#'   Whether to overwrite the primitive if it is already registered.
#' @export
register_primitive <- function(name, primitive, overwrite = FALSE) {
  p <- prim_dict[[name]]
  if (!is.null(p) && !overwrite) {
    cli_abort("Primitive {.field {name}} already registered")
  }
  prim_dict[[name]] <- primitive
}

#' @title Get a Primitive
#' @description
#' Get a primitive by name.
#' @param name (`character()` | `NULL`)\cr
#'   The name of the primitive.
#'   If `NULL`, returns a list of all primitives.
#' @return (`AnvilPrimitive`)
#' @export
prim <- function(name = NULL) {
  if (is.null(name)) {
    return(as.list(prim_dict))
  }
  prim_dict[[name]]
}

is_higher_order_primitive <- function(x) {
  isTRUE(x$higher_order)
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
#' @param register (`logical(1)`)\cr
#'   If `TRUE` (default), register the result under `name` in the primitive
#'   registry.
#' @return A callable of class `c("JitPrimitive", "JitFunction")`.
#' @export
new_primitive <- function(name, fn, subgraphs = character(),
                          static = character(), register = TRUE) {
  checkmate::assert_string(name)
  checkmate::assert_function(fn)
  checkmate::assert_character(subgraphs)
  checkmate::assert_flag(register)

  primitive <- AnvilPrimitive(name, subgraphs = subgraphs)
  jit_fn <- jit(fn, static = static, backend = "auto")
  attr(jit_fn, "primitive") <- primitive
  class(jit_fn) <- c("JitPrimitive", class(jit_fn))

  if (register) {
    assign(name, jit_fn, envir = prim_dict)
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
  if (!is_higher_order_primitive(call$primitive)) {
    return(list())
  }

  stats::setNames(
    lapply(call$primitive$subgraphs, \(sg) {
      call$params[[sg]]
    }),
    call$primitive$subgraphs
  )
}
