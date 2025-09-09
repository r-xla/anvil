hash <- S7::new_generic("hash", "x", function(x) {
  S7::S7_dispatch()
})

method(hash, class_environment) <- function(x) {
  # TODO: Make this nicer
  format(x)
}

# set utils
set <- function() {
  hashtab()
}

set_has <- function(set, key) {
  is.na(gethash(set, key, NA))
}

set_add <- function(set, key) {
  set[[key]] <- NULL
}

get_aval <- function(x) {
  if (inherits(x, Box)) {
    x@aval
  } else if (typeof(x) %in% nvl_types) {
    if (inherits(x, "nvl_array")) {
      stop("currently not supported")
    }
    # TODO: Converter from other types to nvl_array
    ConcreteArray(x)
  } else {
    stop("Type has no aval")
  }
}

dtype_from_buffer <- function(x) {
  d <- as.character(dtype(x))
  switch(
    d,
    pred = stablehlo::BooleanType(),
    i8 = stablehlo::IntegerType("i8"),
    i16 = stablehlo::IntegerType("i16"),
    i32 = stablehlo::IntegerType("i32"),
    i64 = stablehlo::IntegerType("i64"),
    ui8 = stablehlo::IntegerType("ui8"),
    ui16 = stablehlo::IntegerType("ui16"),
    ui32 = stablehlo::IntegerType("ui32"),
    ui64 = stablehlo::IntegerType("ui64"),
    f32 = stablehlo::FloatType("f32"),
    f64 = stablehlo::FloatType("f64"),
    stop("Unsupported dtype: ", d)
  )
}

raise_to_shaped <- function(aval) {
  ShapedArray(aval@dtype, aval@shape)
}


id <- S7::new_generic("id", "x", function(x) {
  S7::S7_dispatch()
})

method(id, class_environment) <- function(x) {
  rlang::addr_address(x)
}

hashkeys <- function(h) {
  val <- vector("list", numhash(h))
  idx <- 0
  maphash(h, function(k, v) {
    idx <<- idx + 1
    val[idx] <<- list(k)
  })
  val
}

is_nvl_type <- function(x) {
  any(sapply(globals$nvl_types, function(type, x) inherits(x, type), x))
}
