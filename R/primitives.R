# Core Primitives

Primitive <- new_class("Primitive",
  properties = list(
    .rules = new_property(class_environment),
    ir_rule = new_property(class_function,
      setter = function(self, value) {
        self@.rules[["ir_rule"]] <- value
        self
      },
      getter = function(self) {
        self@.rules[["ir_rule"]]
      }
    ),
    jit_rule = S7::new_property(class_function,
      setter = function(self, value) {
        self@.rules[["jit_rule"]] <- value
        self
      },
      getter = function(self) {
        self@.rules[["jit_rule"]]
      }
    ),
    backward_rule = S7::new_property(class_function,
      setter = function(self, value) {
        self@.rules[["backward_rule"]] <- value
        self
      },
      getter = function(self) {
        self@.rules[["backward_rule"]]
      }
    )
  ),
  constructor = function() {
    new_object(S7_object(), .rules = new.env())
  }
)

method(print, Primitive) <- function(x) {
  cat(sprintf("<%s>\n", class(x)[[1]]))
}

register_ir_rule <- function(primitive, rule) {
  primitive@ir_rule <- rule
  primitive
}

register_jit_rule <- function(primitive, rule) {
  primitive@jit_rule <- rule
  primitive
}

register_backward_rule <- function(primitive, rule) {
  primitive@backward_rule <- rule
  primitive
}

new_primitive <- function(name) {
  S7::new_class(paste0("Primitive", name), parent = Primitive)()
}

prim_add <- new_primitive("Add")
