.onLoad <- function(libname, pkgname) {
  # FIXME(hack): I don't understand why this is needed
  S7::methods_register()

  ns <- asNamespace(pkgname)
  for (name in ls(ns, pattern = "^p_")) {
    primitive <- get(name, envir = ns)
    register_primitive(sub("^p_", "", name), primitive)
  }

  # fmt: skip
  globals$ranges_raw <- list(
    ui8  = minmax_raw(8, FALSE),
    ui16 = minmax_raw(16, FALSE),
    ui32 = minmax_raw(32, FALSE),
    ui64 = minmax_raw(64, FALSE),
    i8   = minmax_raw(8, TRUE),
    i16  = minmax_raw(16, TRUE),
    i32  = minmax_raw(32, TRUE),
    i64  = minmax_raw(64, TRUE)
  )
}
