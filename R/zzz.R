.onLoad <- function(libname, pkgname) {
  # FIXME(hack): I don't understand why this is needed
  registerS3method("print", "AnvilTensor", print.AnvilTensor)
  S7::methods_register()
}
