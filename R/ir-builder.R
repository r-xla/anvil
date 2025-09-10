#' Here we need reference semantic, because this will be a global variable.
#' The problem is that we can't have one builder always at the bottom of the stack
#' Because
#' @export
Builder <- R6::R6Class(
  "Builder",
  public = list(
    equations = NULL,
    boxes_to_variables = NULL,
    const_boxes = NULL,
    const_variables = NULL,
    boxes = NULL,
    initialize = function() {
      self$equations <- list()
      self$boxes_to_variables <- hashtab()
      self$const_boxes <- hashtab()
      self$const_variables <- hashtab()
      self$boxes <- list()
    },
    add_equation = function(equation) {
      self$equations <- c(self$equations, equation)
      invisible(self)
    },
    add_variable = function(box) {
      self$boxes_to_variables[[id(box)]] <- IRVariable(box@aval)
      invisible(self)
    },
    get_variable = function(box) {
      var <- self$boxes_to_variables[[id(box)]]
      if (is.null(var)) {
        stop("box not found")
      }
      var
    },
    add_constant = function(box, value) {
      self$const_boxes[[id(box)]] <- box
      self$const_variables[[box]] <- value
      invisible(self)
    },
    build = function(in_boxes, out_boxes) {
      const_variables <- hashkeys(self$const_variables)
      const_boxes <- hashkeys(self$const_boxes)
      t2v <- function(x) {
        self$boxes_to_variables[[id(x)]]
      }
      in_binders <- c(const_variables, lapply(in_boxes, t2v))
      out_vars <- lapply(out_boxes, t2v)
      ir <- IR(in_binders, self$equations, out_vars)
      #typecheck_ir(ir)
      ir
    },
    new_box = function(interpreter, aval) {
      if (!inherits(aval, ShapedTensor)) {
        stop("aval must be a ShapedTensor")
      }
      box <- IRBox(
        interpreter = interpreter,
        aval = aval
      )
      self$boxes <- c(self$boxes, box)
      return(box)
    }
  )
)

merge_builders <- function(builders) {
  if (length(builders) == 1) {
    return(builders[[1]])
  }
  merge <- function(x, y) {
    if (identical(x, y)) {
      return(x)
    }
  }

  Builder$new(
    inputs = merge_builder_inputs(builders)
  )
}

merge_builder_inputs <- function(builders) {}

merge_builder_outputs <- function(builders) {}

merge_builder_equations <- function(builders) {}

merge_builder_boxes_to_variables <- function(builders) {}

merge_builder_const_boxes <- function(builders) {}
