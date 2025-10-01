## TODO: Move into helper package (code is currently in stablehlo and this package)
#list_of <- new_class("list_of")
#
#method(`==`, list(list_of, list_of)) <- function(e1, e2) {
#  length(e1@items) == length(e2@items) &&
#    all(
#      sapply(seq_along(e1@items), function(i) {
#        e1@items[[i]] == e2@items[[i]]
#      })
#    )
#}
#
#method(`!=`, list(list_of, list_of)) <- function(e1, e2) {
#  !(e1 == e2)
#}
#
#method(length, list_of) <- function(x) {
#  length(x@items)
#}
#
#new_list_of <- function(class_name, item_type) {
#  new_class(
#    class_name,
#    parent = list_of,
#    properties = list(
#      items = new_property(
#        S7::class_list,
#        validator = function(value) {
#          if (!is.list(value)) {
#            return("Not a list")
#          }
#          if (inherits(item_type, "S7_union")) {
#            item_types <- item_type$classes
#            for (i in seq_along(value)) {
#              for (it in item_types) {
#                ok <- FALSE
#                if (S7:::class_inherits(value[[i]], it)) {
#                  ok <- TRUE
#                  break
#                }
#              }
#              if (!ok) {
#                classnames <- sapply(item_types, \(x) attr(x, "name"))
#                return(sprintf(
#                  "Expected item to be of type %s. Got %s.",
#                  classnames,
#                  class(value[[i]])[[1L]]
#                ))
#              }
#            }
#          } else {
#            for (i in seq_along(value)) {
#              if (!S7:::class_inherits(value[[i]], item_type)) {
#                return(
#                  sprintf(
#                    "Expected item to be of type %s. Got %s.",
#                    S7::S7_class(value[[i]]),
#                    S7::S7_class(item_type)
#                  )
#                )
#              }
#            }
#          }
#        }
#      )
#    )
#  )
#}

list_of <- function(class, ...) {
  new_list_property(of = class, ...)
}

new_list_property <- function(
  ...,
  validator = NULL,
  default = if (isTRUE(named)) {
    quote(setNames(list(), character()))
  } else {
    quote(list())
  },
  of = class_any,
  named = NA,
  min_length = 0L,
  max_length = Inf
) {
  prop <- new_property(
    class_list,
    ...,
    validator = function(value) {
      c(
        if (
          !identical(of, class_any) &&
            !all(vapply(value, S7:::class_inherits, logical(1L), of))
        ) {
          paste("must only contain elements of class", S7:::class_desc(of))
        },
        if (!is.null(of_validator)) {
          msgs <- unlist(lapply(value, of_validator))
          if (length(msgs) > 0L) {
            paste(
              "element(s) failed validation:",
              paste0("'", unique(msgs), "'", collapse = ", ")
            )
          }
        },
        if (isTRUE(named) && is.null(names(value))) {
          "must have names"
        },
        if (identical(named, FALSE) && !is.null(names(value))) {
          "must not have names"
        },
        if (length(value) < min_length || length(value) > max_length) {
          paste0("must have length in [", min_length, ", ", max_length, "]")
        },
        if (!is.null(validator)) {
          validator(value)
        }
      )
    },
    default = default
  )
  prop$of <- of
  if (inherits(of, "S7_property")) {
    of_validator <- of$validator
    of <- of$class
  } else {
    of_validator <- NULL
  }
  prop$named <- named
  class(prop) <- c("list_S7_property", class(prop))
  prop
}
