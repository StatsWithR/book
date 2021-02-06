# Inference and Decision-Making with Multiple Parameters

We saw in \@ref(sec:normal-normal) that if the data followed a normal distribution and that the variance was known, that the normal distribution was the conjugate prior distribution for the unknown  mean. In this chapter, we will focus on the situation when the data follow a normal distribution with an unknown mean, but now consider the case where the variance is also unknown.  When the variance $\sigma^2$ of the data is also unknown, we need to specify a joint prior distribution $p(\mu, \sigma^2)$ for both the mean $\mu$ and the variance $\sigma^2$. We will introduce the conjugate normal-gamma family of distributions where the posterior distribution is in the same family as the prior distribution and leads to a marginal Student t distribution for posterior inference for the mean of the population.

We will present Monte Carlo simulation for inference about functions of the parameters as well as sampling from predictive distributions, which can also be used to assist with prior elicitation. For situations when limited prior information is available, we discuss a limiting case of the normal-gamma conjugate family, the reference prior, leading to a prior that can be used for a default or reference analysis.  Finally, we will show how to create a more flexible and robust prior distribution by using  mixtures of the normal-gamma conjugate prior, the Jeffreys-Zellner-Siow prior. For inference in this case we will introduce Markov Chain Monte Carlo, a powerful simulation method for Bayesian inference.

It is assumed that the readers have mastered the concepts of one-parameter normal-normal conjugate priors. Calculus is not required for this section; however, for those who are comfortable with calculus and would like to go deeper, we shall present optional sections with more details on the derivations.

Also note that some of the examples in this section use an experimental update to the `bayes_inference` function that are not provided in the CRAN released version of the package. To get the experimental version you can install the `BayesFactor` branch from GitHub using the following command in R,

```r
remotes::install_github("statswithr/statsr@BayesFactor")
```
If your local output differents from what is seen in this chapter, or the provided code fails to run for you this is the most likely cause.
