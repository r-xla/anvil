# Pretty printing (functional style)

# A PPrint is represented as a list of lines, each line being a list(indent, s)
pp_new <- function(lines) {
  structure(lines, class = "anvil_pprint")
}

pp <- function(x) {
  xs <- strsplit(as.character(x), "\n", fixed = TRUE)[[1]]
  pp_new(lapply(xs, function(s) list(indent = 0L, s = s)))
}

pp_indent <- function(p, indent) {
  pp_new(lapply(p, function(ls) {
    list(indent = ls$indent + as.integer(indent), s = ls$s)
  }))
}

pp_concat <- function(a, b) {
  pp_new(c(a, b))
}

# Horizontal composition: place b after the last line of a
pp_hcat <- function(a, b) {
  if (length(b) == 0L) {
    return(pp_new(a))
  }
  if (length(a) == 0L) {
    return(pp_new(b))
  }
  indent <- a[[length(a)]]$indent
  s <- a[[length(a)]]$s
  b_ind <- pp_indent(b, indent + nchar(s, type = "width"))
  common_line <- paste0(s, strrep(" ", b[[1]]$indent), b[[1]]$s)
  pp_new(c(
    a[-length(a)],
    list(list(indent = indent, s = common_line)),
    b_ind[-1]
  ))
}

pp_vcat <- function(ps) {
  Reduce(pp_concat, ps, init = pp(""))
}

pp_to_string <- function(p) {
  paste(
    vapply(p, function(ls) paste0(strrep(" ", ls$indent), ls$s), character(1)),
    collapse = "\n"
  )
}

# IRVariable naming utilities
make_name_generator <- function() {
  i <- 0L
  function() {
    i <<- i + 1L
    n <- i
    out <- character()
    while (n > 0L) {
      n <- n - 1L
      r <- n %% 26L
      out <- c(letters[r + 1L], out)
      n <- n %/% 26L
    }
    paste0(out, collapse = "")
  }
}

get_var_name <- function(names_env, namegen, v) {
  key <- id(v@.id)
  nm <- names_env[[key]]
  if (is.null(nm)) {
    nm <- namegen()
    names_env[[key]] <- nm
  }
  nm
}

var_str <- function(names_env, namegen, v) {
  sprintf("%s:%s", get_var_name(names_env, namegen, v), repr(v@aval))
}

# IRParams pretty printer (sorted by name when available)
pp_params <- function(params) {
  if (length(params)) {
    nms <- names(params)
    if (!is.null(nms)) {
      ord <- order(nms)
      items <- mapply(
        function(k, v) sprintf("%s=%s", k, toString(v)),
        nms[ord],
        params[ord],
        SIMPLIFY = TRUE,
        USE.NAMES = FALSE
      )
    } else {
      items <- vapply(params, toString, character(1))
    }
    pp(" [ ") |>
      pp_hcat(pp_vcat(lapply(as.list(items), pp))) |>
      pp_hcat(pp(" ] "))
  } else {
    pp(" ")
  }
}

# Custom pretty-print rules hook (by primitive name)
pp_rules <- new.env(parent = emptyenv())

pp_atom <- function(names_env, namegen, x) {
  if (inherits(x, IRVariable)) {
    get_var_name(names_env, namegen, x)
  } else if (inherits(x, IRLiteral)) {
    sprintf("const<%s>", repr(x@aval))
  } else {
    stop("Unknown atom type for pretty printing")
  }
}

pp_eqn <- function(names_env, namegen, eqn) {
  rule <- pp_rules[[eqn@primitive@name]]
  if (!is.null(rule)) {
    return(rule(names_env, namegen, eqn))
  }
  lhs <- pp(paste(
    vapply(
      eqn@out_binders,
      function(v) var_str(names_env, namegen, v),
      character(1)
    ),
    collapse = " "
  ))
  inputs <- paste(
    vapply(
      eqn@inputs,
      function(x) pp_atom(names_env, namegen, x),
      character(1)
    ),
    collapse = " "
  )
  rhs <- pp(eqn@primitive@name) |>
    pp_hcat(pp_params(eqn@params)) |>
    pp_hcat(pp(inputs))
  lhs |> pp_hcat(pp(" = ")) |> pp_hcat(rhs)
}

pp_ir <- function(ir) {
  namegen <- make_name_generator()
  names_env <- new.env(parent = emptyenv())
  in_binders <- paste(
    vapply(
      ir@in_binders,
      function(v) var_str(names_env, namegen, v),
      character(1)
    ),
    collapse = ", "
  )
  eqns <- pp_vcat(lapply(ir@equations, function(e) {
    pp_eqn(names_env, namegen, e)
  }))
  outs <- paste(
    vapply(
      ir@outputs,
      function(v) get_var_name(names_env, namegen, v),
      character(1)
    ),
    collapse = ", "
  )
  header <- pp("{ lambda ") |> pp_hcat(pp(in_binders)) |> pp_hcat(pp(" ."))
  body <- (pp("let ") |> pp_hcat(eqns)) |>
    pp_concat(pp(paste0("in ( ", outs, " ) }")))
  pp_indent(pp_concat(header, body), 2L)
}

method(repr, IR) <- function(x) {
  pp_to_string(pp_ir(x))
}
