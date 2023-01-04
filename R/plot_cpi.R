library(ggplot2)


plot_surv_cpi <- function(matrix.cpi) {
  theme_set(theme_bw())

  ggplot(data = matrix.cpi, 
         aes(x = Times, y = CPI, group = Variable, color = Variable)) +
    geom_line() +
    ggtitle("Time-Dependent Conditional Predictive Importance") +
    xlab("") +
    theme(plot.title = element_text(hjust = 0.5))}





