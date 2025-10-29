.onLoad <- function(libname, pkgname) {
  # FIXME(hack): I don't understand why this is needed
  S7::methods_register()

  # fmt: skip
  globals$ranges_raw <- list(
    ui8  = minmax_raw(8, FALSE),
    ui16 = minmax_raw(8, FALSE),
    ui32 = minmax_raw(8, FALSE),
    u64  = minmax_raw(8, FALSE),
    i8   = minmax_raw(8, TRUE),
    i16  = minmax_raw(8, TRUE),
    i32  = minmax_raw(8, TRUE),
    i64  = minmax_raw(8, TRUE)
  )
}
