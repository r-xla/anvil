#' <% p <- getFromNamespace("prim", "anvil")(primitive_id); implemented <- Filter(function(r) !is.null(p$rules[[r]]), getFromNamespace("globals", "anvil")$interpretation_rules) %>
#' <% if (length(implemented) > 0) { %>
#' @section Implemented Rules:
#' <%= paste0("- `", implemented, "`", collapse = "\n#' ") %>
#' <% } %>
