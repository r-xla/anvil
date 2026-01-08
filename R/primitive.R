#' @title AnvilPrimitive
#' @description
#' Primitive interpretation rule.
#' @param name (`character()`)\cr
#'   The name of the primitive.
#' @return (`AnvilPrimitive`)
#' @export
AnvilPrimitive <- function(name) {
  checkmate::assert_string(name)
  env <- new.env(parent = emptyenv())
  env$name <- name
  env$rules <- list()

  structure(env, class = "AnvilPrimitive")
}

#' @export
`$.AnvilPrimitive` <- function(x, name) {
  x[[name]]
}

#' @export
`$<-.AnvilPrimitive` <- function(x, name, value) {
  x[[name]] <- value
  x
}

#' @title HigherOrderPrimitive
#' @description
#' A primitive that contains subgraphs.
#' @param name (`character()`)\cr
#'   The name of the primitive.
#' @param subgraphs (`character()`)\cr
#'   Names of parameters that are subgraphs.
#' @return (`HigherOrderPrimitive`)
#' @export
HigherOrderPrimitive <- function(name, subgraphs = character()) {
  checkmate::assert_string(name)
  checkmate::assert_character(subgraphs)

  env <- new.env(parent = emptyenv())
  env$name <- name
  env$rules <- list()
  env$subgraphs <- subgraphs

  structure(env, class = c("HigherOrderPrimitive", "AnvilPrimitive"))
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
  inherits(x, "HigherOrderPrimitive")
}


#' @method [[<- AnvilPrimitive
#' @export
`[[<-.AnvilPrimitive` <- function(x, name, value) {
  if (name %in% globals$interpretation_rules) {
    # Store interpretation rule in the rules list
    rules <- get("rules", envir = x, inherits = FALSE)
    rules[[name]] <- value
    assign("rules", rules, envir = x)
  } else {
    # Store other properties directly in environment
    assign(name, value, envir = x)
  }
  x
}

#' @method [[ AnvilPrimitive
#' @export
`[[.AnvilPrimitive` <- function(x, name) {
  if (name %in% globals$interpretation_rules) {
    # Access interpretation rule from rules list
    rules <- get("rules", envir = x, inherits = FALSE)
    rule <- rules[[name]]
    if (is.null(rule)) {
      cli_abort("Rule {.field {name}} not defined for primitive {.field {x$name}}")
    }
    return(rule)
  }
  # Access other properties from environment
  get(name, envir = x, inherits = FALSE)
}

#' @method print AnvilPrimitive
#' @export
print.AnvilPrimitive <- function(x, ...) {
  cat(sprintf("<AnvilPrimitive:%s>\n", x$name))
  invisible(x)
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

#' @rdname AnvilPrimitive
#' @export
Primitive <- AnvilPrimitive
