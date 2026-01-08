#' @title AnvilPrimitive
#' @description
#' Primitive interpretation rule.
#' @param name (`character()`)\cr
#'   The name of the primitive.
#' @return (`AnvilPrimitive`)
#' @export
AnvilPrimitive <- function(name) {
  checkmate::assert_string(name)
  env <- zero_env()

  structure(
    list(name = name, rules = env),
    class = "AnvilPrimitive"
  )
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

  structure(
    list(name = name, rules = zero_env(), subgraphs = subgraphs),
    class = c("HigherOrderPrimitive", "AnvilPrimitive")
  )
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
  x$rules[[name]] <- value
  if (!(name %in% globals$interpretation_rules)) {
    cli_abort("Unknown interpretation rule: {.val {name}}")
  }
  x
}

#' @method [[ AnvilPrimitive
#' @export
`[[.AnvilPrimitive` <- function(x, name) {
  rule <- x$rules[[name]]
  if (is.null(rule)) {
    if (!(name %in% globals$interpretation_rules)) {
      cli_abort("Unknown rule: {name}")
    }
    cli_abort("Rule {.field {name}} not defined for primitive {.field {x$name}}")
  }
  rule
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
