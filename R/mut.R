# copied from: https://codeberg.org/dgkf/S7.mut/src/branch/main/R/mut.R
# TODO: Attribution

#' @import S7
NULL

.class_desc <- function(...) {
  getNamespace("S7")[["class_desc"]](...)
}

mut_constructors <- new.env(parent = baseenv())

mut_of <- function(x) {
  get0(.class_desc(x), envir = mut_constructors)
}

mut_getter <- function(fn, name, .state = ".state") {
  name

  if (is.null(fn)) {
    fn <- function(self) {
      prop(self, .state)[[name]]
    }
  }

  fn
}

mut_setter <- function(fn, name, .state = ".state") {
  name

  if (is.null(fn)) {
    fn <- function(self, value) {
      prop(self, .state)[[name]] <- prop(self, name) <- value
      attr(self, name) <- NULL
      self
    }
  } else {
    body(fn) <- bquote({
      .(body(fn))
      prop(self, .state)[[name]] <- prop(self, name)
      attr(self, name) <- NULL
      self
    })
  }

  fn
}

as_mut_S7_object <- function(x, .state = ".state") {
  # nolint: object_name_linter, line_length_linter.
  # initialize state, since we don't include it inour constructor
  default_state <- attr(x, "S7_class")@properties[[.state]]$default
  prop(x, .state) <- if (is.language(default_state)) {
    eval.parent(default_state)
  } else {
    default_state
  }

  # add a S3 class
  class(x) <- append(
    class(x),
    after = which(class(x) == "S7_object") - 1L,
    "S7_mut_object"
  )

  x
}

#' Convert an `S7` class to a mutable `S7` object
#'
#' @param x `S7_class` an `S7_class` class constructor - the function you
#'   would otherwise call to create a new object of that class.
#'
#' @examples
#' library(S7)
#'
#' class_ex <- new_class(
#'   "class_example",
#'   properties = list(
#'     # validators
#'     value = new_property(
#'       class = class_integer,
#'       validator = function(value) {
#'         if (value == 0L) "value cannot be exactly 0"
#'       }
#'     ),
#'     # read-only properties
#'     is_gt_0 = new_property(
#'       class = class_logical,
#'       getter = function(self) {
#'         self@value > 0
#'       }
#'     )
#'   )
#' )
#'
#' # make a mutable version of our class
#' ex <- mut(class_ex)(value = 3L)
#' ex@value
#'
#' # we can make a copy and update our value property
#' ex_ref <- ex
#' ex_ref@value <- 30L
#'
#' # all values reference the same data, our original is updated
#' ex@value
#'
#' @export
mut <- function(x) {
  # if we've already built a mutable equivalent of x, return it
  if (!is.null(mut_constructor <- mut_of(x))) {
    return(mut_constructor)
  }

  # otherwise, build it, save it and return it
  args <- S7::props(x)
  args$name <- paste0("mut<", args$name, ">")

  # build our property definitions for mutable wrapper
  for (prop in args$properties) {
    # avoid creating setter for read-only properties
    if (!is.null(prop$setter) || is.null(prop$getter)) {
      prop$setter <- mut_setter(prop$setter, prop$name)
    }

    prop$getter <- mut_getter(prop$getter, prop$name)
    args$properties[[prop$name]] <- prop
  }

  # add our inner mutable state as a property
  args$properties$.state <- new_property(
    class = class_environment,
    getter = function(self) {
      # initialize on first use; default doesn't work
      if (is.null(self@.state)) {
        self@.state <- new.env(parent = baseenv())
      }

      self@.state
    },
    setter = function(self, value) {
      if (is.environment(value)) {
        self@.state <- value
      }
      self
    }
  )

  # build our class
  cls <- do.call(S7::new_class, args)

  # update constructor to flag objects as mutable
  attrs <- attributes(cls)

  const_env <- new.env()
  const_env$as_mut_S7_object <- as_mut_S7_object
  parent.env(const_env) <- environment(attrs$constructor)
  environment(cls) <- environment(attrs$constructor) <- const_env
  body(cls) <- body(attrs$constructor) <- bquote({
    obj <- .(body(cls@constructor))
    as_mut_S7_object(obj)
  })

  attributes(cls) <- attrs

  # cache our class constructor so that we don't need to re-build it
  mut_constructors[[.class_desc(cls)]] <- cls

  cls
}

clone <- new_generic("clone", "x")

method(clone, new_S3_class("S7_mut_object")) <- function(x) {
  cloned_state <- new.env(parent = parent.env(x@.state))

  for (name in names(x@.state)) {
    cloned_state[[name]] <- prop(x, name)
  }

  prop(x, ".state") <- cloned_state

  x
}
