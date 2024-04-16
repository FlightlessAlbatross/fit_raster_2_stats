


Replace_Values_Map <- function(x, dictionary) {
  
  # replace all instances of multiple values with other values according to a dictionary
  x.f <- factor(x)
  
  dictionary <- dictionary[dictionary$original %in% x,]
  
  levels(x.f)[match(dictionary$original, levels(x.f))] <- dictionary$new
  
  return(as.character(x.f))

  }

if(F){
  dictionary <- data.frame(
    "original" = c("WBG", "ROM", "UTZ"),
    "new"      = c("PSE", "ROU", "UTZ"))
  
  x <- c("AUT", "ALB", "WBG", "ROM")
  
  
  Replace_Values_Map (x = x, dictionary = dictionary)
  
  
}
