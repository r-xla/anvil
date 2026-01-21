# shamelessly copied from: https://github.com/tidyverse/readr/blob/e529cb2775f1b52a0dfa30dabc9f8e0014aa77e6/R/zzz.R
register_s3_method <- function(pkg, generic, class, fun = NULL) {
  if (is.null(fun)) {
    fun <- get(paste0(generic, ".", class), envir = parent.frame())
  } else {
    stopifnot(is.function(fun))
  }

  if (pkg %in% loadedNamespaces()) {
    registerS3method(generic, class, fun, envir = asNamespace(pkg))
  }

  # Always register hook in case package is later unloaded & reloaded
  setHook(
    packageEvent(pkg, "onLoad"),
    function(...) {
      registerS3method(generic, class, fun, envir = asNamespace(pkg))
    },
    action = "append"
  )
}

register_namespace_callback <- function(pkgname, namespace, callback) {
  assert_string(pkgname)
  assert_string(namespace)
  assert_function(callback)

  remove_hook <- function(event) {
    hooks <- getHook(event)
    pkgnames <- vapply(
      hooks,
      function(x) {
        ee <- environment(x)
        if (isNamespace(ee)) environmentName(ee) else environment(x)$pkgname
      },
      NA_character_
    )
    setHook(event, hooks[pkgnames != pkgname], action = "replace")
  }

  remove_hooks <- function(...) {
    remove_hook(packageEvent(namespace, "onLoad"))
    remove_hook(packageEvent(pkgname, "onUnload"))
  }

  if (isNamespaceLoaded(namespace)) {
    callback()
  }

  setHook(packageEvent(namespace, "onLoad"), callback, action = "append")
  setHook(packageEvent(pkgname, "onUnload"), remove_hooks, action = "append")
}

.onLoad <- function(libname, pkgname) {
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

  # Register compare_proxy for waldo/testthat
  register_s3_method("waldo", "compare_proxy", "AnvilTensor")
}
