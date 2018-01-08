---
output:
  pdf_document: default
  html_document: default
---
## Bayesian Multiple Linear Regression

In this section, we will discuss Bayesian inference in multiple linear regression. We will use the reference prior to provide the default, or base line analysis of the model, which provides the correspondence between Bayesian and frequentist approaches.

### The Model

To illustrate the idea, we use the data set that we examined earlier on kids' cognitive scores, where we predicted the value of the kid's cognitive score from the mother's high school status, mother's IQ score, whether or not the mother worked during the first three years of the kid's life, and the mother's age. We set up the model as follows
$$ \text{score}_i = \beta_0 + \beta_1 \text{hs}_i + \beta_2\text{IQ}_i + \beta_3\text{work}_i + \beta_4 \text{age}_i + \epsilon_i. $$

Here, $\text{score}_i$ is the $i$th kid's cognitive score. $\text{hs}_i$, $\text{IQ}_i$, $\text{work}_i$, and $\text{age}_i$ represent the high school status, the IQ score, work status during the first three years and the age of the $i$th kid's mother. $\epsilon_i$ is the error term. 

### Data Pre-processing

We can download the data set from Gelman's website and read the summary information of the data set.

```r
cognitive = foreign::read.dta("http://www.stat.columbia.edu/~gelman/arm/examples/child.iq/kidiq.dta")
summary(cognitive)
```

```
##    kid_score         mom_hs           mom_iq          mom_work    
##  Min.   : 20.0   Min.   :0.0000   Min.   : 71.04   Min.   :1.000  
##  1st Qu.: 74.0   1st Qu.:1.0000   1st Qu.: 88.66   1st Qu.:2.000  
##  Median : 90.0   Median :1.0000   Median : 97.92   Median :3.000  
##  Mean   : 86.8   Mean   :0.7857   Mean   :100.00   Mean   :2.896  
##  3rd Qu.:102.0   3rd Qu.:1.0000   3rd Qu.:110.27   3rd Qu.:4.000  
##  Max.   :144.0   Max.   :1.0000   Max.   :138.89   Max.   :4.000  
##     mom_age     
##  Min.   :17.00  
##  1st Qu.:21.00  
##  Median :23.00  
##  Mean   :22.79  
##  3rd Qu.:25.00  
##  Max.   :29.00
```

As we see that variables `mom_hs` and `mom_work` should be considered as categorical variables. We transform them into indicator variables where `mom_work` shows whether the mother worked for 1 or more years, and `mom_hs` indicates whether the mother had more than a high school education.

The code is as below:^[Note: `as.numeric` is not necessary here. We use `as.numeric` to keep the names of the levels of the two variables short.]


```r
cognitive$mom_work <- as.numeric(cognitive$mom_work > 1)
cognitive$mom_hs <- as.numeric(cognitive$mom_hs > 0)

# Modify column names of the data set
colnames(cognitive) <- c("kid_score", "hs", "IQ", "work", "age")
```

### Specify Bayesian Prior Distributions

For Bayesian inference, we need to specify a prior distribution of the error terms. Since the kid's cognitive scores $\text{score}_i$ are continuous, we assume that $\epsilon_i$ are independent, and identically distributed with the normal distribution
$$ \epsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2), $$
where $\sigma^2$ is the commonly shared variance of all observations.

We will also need to specify the prior distribution for all the coefficients $\beta_0,\ \beta_1,\ \beta_2,\ \beta_3$, and $\beta_4$. An informative prior, which assumes that the $\beta$'s follow the multivariate normal distribution with covariance matrix $\sigma^2\Sigma_0$ can be used. We may further impose the inverse Gamma distribution to $\sigma^2$, to complete the hierachical model
$$ 
\begin{aligned}
\beta_0, \beta_1, \beta_2, \beta_3, \beta_4 ~|~\sigma^2 \ \sim & \mathcal{N}((b_0, b_1, b_2, b_3, b_4), \sigma^2\Sigma_0)\\
\sigma^2 \ \sim & \text{IG}(\nu_0/2, \nu_0\sigma_0^2/2) 
\end{aligned}
$$

This gives us the multivariate normal-gamma conjugate family, with hyperparameters $b_0, b_1, b_2, b_3, b_4, \Sigma_0, \nu_0$, and $\sigma_0^2$. For this prior, we will need to specify the values of all the hyperparameters. This elicitation can be quite involved, especially when we do not have enough prior information about the variances, covariances of the coefficients and other prior hyperparameters. Therefore, we are going to adopt the noninformative prior (i.e., the reference prior), which is a limiting case of this multivariate normal-gamma prior.

The reference prior in the multiple linear regression model is similar to the reference prior we used in the simple linear regression model, where the prior distribution of all the coefficients $\beta$'s is the uniform prior, and the prior of $\sigma^2$ is proportional to its reciprocal
$$ \pi(\beta_0,\beta_1,\beta_2,\beta_3,\beta_4~|~\sigma^2) \propto 1,\qquad \pi(\sigma^2) \propto \frac{1}{\sigma^2}. $$

Under this reference prior, the posterior distributions of the coefficients, $\beta$'s, are parallel to the ones in simple linear regression. The marginal posterior distributions of $\beta$'s are the Student's $t$-distributions with centers given by the frequentist OLS estimates  $\hat{\beta}$'s, scale parameters given by the standard errors $\text{SE}_{\hat{\beta}}^2$ obtained from the OLS estimates
$$
\beta_j~|~y_1,\cdots,y_n\ \sim t_{n-p-1}(\hat{\beta}_j, \text{SE}_{\hat{\beta}_j}^2),\qquad j = 0, 1, \cdots, p.
$$

The degrees of freedom of these $t$-distributions are $n-p-1$, where $p$ is the number of predictors. In the kid cognitive score example, $p=4$. The posterior mean, $\hat{\beta}_j$, is the center of the $t$-distributions of $\beta_j$, which is the same as the OLS estimates of $\beta_j$. The posterior standard deviation of $\beta_j$, which is the square root of the scale parameter of the $t$-distribution, is $\text{SE}_{\beta_j}$, the standard error of $\beta_j$ under the OLS estimates. That means, under the reference prior, we can easily obtain the posterior mean and posterior standard deviation from using the `lm` function, since they are numerically equivalent to the counterpart of the frequentist approach.


### Fitting the Bayesian Model

To gain more flexibility in choosing priors, we will instead use the `bas.lm` function in the `BAS` library, which allows us to specify different model priors.


```r
library(BAS)
cog.bas = bas.lm(kid_score ~ ., data = cognitive, prior = "BIC", 
                 modelprior = Bernoulli(1), bestmodel = rep(1, 5), n.models = 1)
```

The above `bas.lm` function uses the model formula the same as in the `lm`. It first specifies the response and predictor variables, a data argument to provide the data frame. The addition arguments further include the prior on the coefficients. We use `"BIC"` here to indicate that the model is based on the non-informative reference prior. (We will explain in the later section why we use the name `"BIC"`.) The `modelprior` argument tells the function to include all variables, which can be viewed as every variable is included with probability 1 under a Bernoulli trial. Because we want to fit just the full model, we use `bestmodel = rep(1,5)` to indicate that the intercept and all 4 predictors are included. The argument `n.models = 1` fit just this one model.


### Posterior Means and Posterior Standard Deviations

Similar to the OLS regression process, we can extract the posterior means and standard deviations of the coefficients using the `coef` function


```r
cog.coef = coef(cog.bas)
cog.coef
```

```
## 
##  Marginal Posterior Summaries of Coefficients: 
## 
##  Using  BMA 
## 
##  Based on the top  1 models 
##            post mean  post SD   post p(B != 0)
## Intercept  86.79724    0.87092   1.00000      
## hs          5.09482    2.31450   1.00000      
## IQ          0.56147    0.06064   1.00000      
## work        2.53718    2.35067   1.00000      
## age         0.21802    0.33074   1.00000
```

From the last column in this summary, we see that the probability of the coefficients to be non-zero is always 1. This is because we specify the argument `Bernoulli(1)` to force the model to include all variables.

We can visualize the coefficients in front of the predictors $\beta_1,\beta_2, \beta_3, \beta_4$ using the `plot` function. We use the `subset` argument to plot only the coefficients of the predictors.


```r
par(mfrow = c(2, 2), col.lab = "darkgrey", col.axis = "darkgrey", col = "darkgrey")
plot(cog.coef, subset = 2:5, ask = F)
```

![](06-regression-03-Bayesian-multi-regression_files/figure-latex/plot-coef-1.pdf)<!-- --> 

These distributions all center at their respetive OLS estimates $\hat{\beta}_j$, with the spread of the distribution related to the standard errors. 

### Credible Intervals Summary

We can also report the posterior means, posterior standard deviations, and the 95% credible intervals of the coefficients of all 4 predictors, which may give a clearer and more useful summary. The `BAS` library provides the method `confint` to extract the credible intervals from the output `cog.coef`. If we are only interested in the distributions of the coefficients of the 4 predictors, we may use the `parm` argument to restrict the variables shown in the summary

```r
confint(cog.coef, parm = 2:5)
```

```
##            2.5%     97.5%      beta
## hs    0.5456507 9.6439990 5.0948248
## IQ    0.4422784 0.6806616 0.5614700
## work -2.0830879 7.1574454 2.5371788
## age  -0.4320547 0.8680925 0.2180189
## attr(,"Probability")
## [1] 0.95
## attr(,"class")
## [1] "confint.bas"
```

All together, we can generate a summary table showing the posterior means, posterior standard deviations, the upper and lower bounds of the 95% credible intervals of all coefficients $\beta_0, \beta_1, \beta_2, \beta_3$, and $\beta_4$.


```r
out <- confint(cog.coef)[, 1:2]  # only extract the upper and lower bounds of the credible intervals
names <- c("posterior mean", "posterior std", colnames(out))
out <- cbind(cog.coef$postmean, cog.coef$postsd, out)
colnames(out) <- names

round(out, 2)
```

```
##           posterior mean posterior std  2.5% 97.5%
## Intercept          86.80          0.87 85.09 88.51
## hs                  5.09          2.31  0.55  9.64
## IQ                  0.56          0.06  0.44  0.68
## work                2.54          2.35 -2.08  7.16
## age                 0.22          0.33 -0.43  0.87
```

As in the simple linear aggression, the posterior estimates from the reference prior, that are in the table, are equivalent to the numbers reported from the `lm` function in R, or using the confident function in the OLS estimates. These intervals are centered at the posterior mean with width given by  the appropriate $t$ quantile with $n-p-1$ degrees of freedom times the posterior standard deviation. The primary difference is the interpretation of the intervals. For example, given this data we believe there is a 95% chance that the kid's cognitive score increases by 0.44 to 0.68 with an additional increase of the mother's IQ score. The mother's high school status has a larger effect where we believe that there is a 95% chance the kid would score of 0.55 up to 9.64 points higher if the mother had three or more years of high school. The credible intervals of the predictors `work` and `age` include 0, which implies that we may improve this model so that the model will accomplish a desired level of explanation or prediction with fewer predictors. 



