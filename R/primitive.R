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
  parent = Primitive,
  properties = list(
    subgraphs = class_character
  ),
  constructor = function(name, subgraphs = character()) {
    obj <- S7::new_object(
      S7::S7_object(),
      name = name,
      rules = zero_env(),
      subgraphs = subgraphs
    )
    obj
  }
)

prim_dict <- new.env(parent = emptyenv())

#' @title Register a Primitive
#' @description
#' Register a primitive.
#' @param name (`character()`)\cr
#'   The name of the primitive.
#' @param primitive (`Primitive`)\cr
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
#' @return (`Primitive`)
#' @export
prim <- function(name = NULL) {
  if (is.null(name)) {
    return(as.list(prim_dict))
  }
  prim_dict[[name]]
}

is_higher_order_primitive <- function(x) {
  inherits(x, "anvil::HigherOrderPrimitive")
}


#' @export
`[[<-.anvil::Primitive` <- function(x, name, value) {
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

#' @title Get Subgraphs from Higher-Order Primitive
#' @description
#' Extracts subgraphs from the parameters of a higher-order primitive call.
#' @param call (`PrimitiveCall`)\cr
#'   The primitive call.
#' @return (`list(Graph)`)\cr
#'   List of subgraphs found in the parameters.
#' @export
subgraphs <- function(call) {
  if (!is_higher_order_primitive(call@primitive)) {
    return(list())
  }

  stats::setNames(
    lapply(call@primitive@subgraphs, \(sg) {
      call@params[[sg]]
    }),
    call@primitive@subgraphs
  )
}
