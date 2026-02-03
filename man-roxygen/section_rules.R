#' <%
#' p <- getFromNamespace(paste0("p_", primitive_id), "anvil")
#' all_rules <- c("stablehlo", "backward")
#' implemented <- Filter(function(r) !is.null(p$rules[[r]]), all_rules)
#' %>
#' @section Interpretation Rules:
#' <%= paste0("- `", implemented, "`", collapse = "\n#' ") %>
