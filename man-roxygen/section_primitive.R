#' @section Access:
#' Access this primitive via `prim$<%= primitive_id %>`.
#' <% p <- prim[[primitive_id]]; implemented <- Filter(function(r) !is.null(p[[r]]), globals$interpretation_rules) %>
#' <% if (length(implemented) > 0) { %>
#' @section Implemented Rules:
#' <%= paste0("- `", implemented, "`", collapse = "\n#' ") %>
#' <% } %>
