# Get the default backend

Returns the current default backend from
`getOption("anvl.default_backend", "xla")`.

## Usage

``` r
default_backend()
```

## Value

`character(1)` — the backend name (e.g. `"xla"`, `"quickr"`).

## See also

[`local_backend()`](https://r-xla.github.io/anvl/reference/local_backend.md)
