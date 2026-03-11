#' @importFrom utils bibentry
bibentries = c( # nolint
  haario_1999 = bibentry("article",
    title   = "Adaptive Proposal Distribution for Random Walk Metropolis Algorithm",
    author  = "Heikki Haario and Eero Saksman and Johanna Tamminen",
    year    = "1999",
    journal = "Computational Statistics",
    volume  = "14",
    number  = "3",
    pages   = "375--395",
    doi     = "10.1007/s001800050022"
  )
)

#' @title Format Bibentries
#'
#' @description
#' Operates on a named list of [bibentry()] entries and formats them for
#' documentation with \CRANpkg{roxygen2}.
#'
#' * `format_bib()` is intended to be called in the `@references` section and
#'   formats the complete entry using [tools::toRd()].
#' * `cite_bib()` returns a short inline citation in the format
#'   `"LastName (Year)"`.
#'
#' @param ... (`character()`)\cr
#'   One or more names of bibentries.
#' @param bibentries (named `list()`)\cr
#'   Named list of bibentries.
#'
#' @return (`character(1)`).
#' @keywords internal
format_bib = function(..., bibentries = NULL) { # nolint
  if (is.null(bibentries)) {
    bibentries = get("bibentries", envir = parent.env(environment()))
  }
  assert_list(bibentries, "bibentry", names = "unique")
  keys = list(...)
  str = vapply(keys, function(entry) tools::toRd(bibentries[[entry]]), character(1))
  paste0(str, collapse = "\n\n")
}

#' @rdname format_bib
#' @keywords internal
cite_bib = function(..., bibentries = NULL) { # nolint
  if (is.null(bibentries)) {
    bibentries = get("bibentries", envir = parent.env(environment()))
  }
  assert_list(bibentries, "bibentry", names = "unique")

  keys = list(...)
  str = vapply(keys, function(entry) {
    x = bibentries[[entry]]
    family = x$author[[1L]]$family
    if (is.null(family)) family = x$author[[1L]]
    sprintf("%s (%s)", family, x$year)
  }, character(1))

  if (length(str) >= 3L) {
    str = c(toString(head(str, -1L)), tail(str, 1L))
  }

  paste0(str, collapse = " and ")
}
