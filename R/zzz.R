.onLoad <- function(libname, pkgname) {
  # FIXME(hack): I don't understand why this is needed
  S7::methods_register()
}
