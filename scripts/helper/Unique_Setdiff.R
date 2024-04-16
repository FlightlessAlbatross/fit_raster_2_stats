# Martin Hofer 2020-05-13


# This #' is a preparation to use roxygen lateron

#' Unique_Setdiff 
#'
#' @param x Object, usually vector or list that is checked for differences
#' @param y Object, usually vector or list that is checked for differences
#'
#' @return A list with two elements, containing the unique objects contained in x, but not in y and vice versa. 
#' @export
#'
#' @examples
#'  # Differnces between IIASA and NODES
#' x <- c("I", "I", "A", "S", "A")
#' y <- c("N", "O", "D", "E", "S")
#' Unique_Setdiff (x, y)






Unique_Setdiff <- function(x,y) {

  x_u <- unique(y)
  y_u <- unique(x)
  
  return(
    list(
      "in x not in y" = setdiff(y_u, x_u),
      "in y not in x" = setdiff(x_u, y_u)
    )  
  )
  
}

