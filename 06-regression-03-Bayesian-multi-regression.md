---
output:
  pdf_document: default
  html_document: default
---
## Bayesian Multiple Linear Regression {#sec:Bayes-multiple-regression}

In this section, we will discuss Bayesian inference in multiple linear regression. We will use the reference prior to provide the default or base line analysis of the model, which provides the correspondence between Bayesian and frequentist approaches.

### The Model

To illustrate the idea, we use the data set on kid's cognitive scores that we examined earlier. We predicted the value of the kid's cognitive score from the mother's high school status, mother's IQ score, whether or not the mother worked during the first three years of the kid's life, and the mother's age. We set up the model as follows

\begin{equation}
y_{\text{score},i} = \alpha + \beta_1 x_{\text{hs},i} + \beta_2 x_{\text{IQ},i} + \beta_3x_{\text{work},i} + \beta_4 x_{\text{age},i} + \epsilon_i, \quad i = 1,\cdots, n.
(\#eq:multi-model1)
\end{equation}

Here, $y_{\text{score},i}$ is the $i$th kid's cognitive score. $x_{\text{hs},i}$, $x_{\text{IQ},i}$, $x_{\text{work},i}$, and $x_{\text{age},i}$ represent the high school status, the IQ score, the work status during the first three years of the kid's life, and the age of the $i$th kid's mother. $\epsilon_i$ is the error term. $n$ denotes the number of observations in this data set.

For better analyses, one usually centers the variable, which ends up getting the following form

\begin{equation} 
y_{\text{score}, i} = \beta_0 + \beta_1 (x_{\text{hs},i}-\bar{x}_{\text{hs}}) + \beta_2 (x_{\text{IQ},i}-\bar{x}_{\text{IQ}}) + \beta_3(x_{\text{work},i}-\bar{x}_{\text{work}}) + \beta_4 (x_{\text{age},i}-\bar{x}_{\text{age}}) + \epsilon_i.
(\#eq:multi-model2)
\end{equation}

Under this tranformation, the coefficients, $\beta_1,\ \beta_2,\ \beta_3$, $\beta_4$, that are in front of the variables, are unchanged compared to the ones in \@ref(eq:multi-model1). However, the constant coefficient $\beta_0$ is no longer the constant coefficient $\alpha$ in \@ref(eq:multi-model1).  Instead, under the assumption that $\epsilon_i$ is independently, identically normal, $\hat{\beta}_0$ is the sample mean of the response variable $Y_{\text{score}}$.^[Under the normal assumption, the mean of the error is 0. Taking mean on both sides of equation \@ref(eq:multi-model2) immediately gives $\beta_0=\bar{y}_{\text{score}}$.] This provides more meaning to $\beta_0$ as this is the mean of $Y$ when each of the predictors is equal to their respective means.  Moreover, it is more convenient to use this "centered" model to derive analyses. The R codes in the `BAS` package are based on the form \@ref(eq:multi-model2).  


### Data Pre-processing

We can download the data set from Gelman's website and read the summary information of the data set using the `read.dta` function in the `foreign` package.

```r
library(foreign)
cognitive = read.dta("http://www.stat.columbia.edu/~gelman/arm/examples/child.iq/kidiq.dta")
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

From the summary statistics, variables `mom_hs` and `mom_work` should be considered as categorical variables. We transform them into indicator variables where `mom_work = 1` if the mother worked for 1 or more years, and `mom_hs = 1` indicates the mother had more than a high school education.

The code is as below:^[Note: `as.numeric` is not necessary here. We use `as.numeric` to keep the names of the levels of the two variables short.]


```r
cognitive$mom_work = as.numeric(cognitive$mom_work > 1)
cognitive$mom_hs = as.numeric(cognitive$mom_hs > 0)

# Modify column names of the data set
colnames(cognitive) = c("kid_score", "hs", "IQ", "work", "age")
```

### Specify Bayesian Prior Distributions

For Bayesian inference, we need to specify a prior distribution for the error term $\epsilon_i$. Since each kid's cognitive score $y_{\text{score},i}$ is continuous, we assume that $\epsilon_i$ is independent, and identically distributed with the Normal distribution
$$ \epsilon_i \iid \No(0, \sigma^2), $$
where $\sigma^2$ is the commonly shared variance of all observations.

We will also need to specify the prior distributions for all the coefficients $\beta_0,\ \beta_1,\ \beta_2,\ \beta_3$, and $\beta_4$. An informative prior, which assumes that the $\beta$'s follow the multivariate normal distribution with covariance matrix $\sigma^2\Sigma_0$ can be used. We may further impose the inverse Gamma distribution to $\sigma^2$, to complete the hierachical model
$$ 
\begin{aligned}
\beta_0, \beta_1, \beta_2, \beta_3, \beta_4 ~|~\sigma^2 ~\sim ~ & \No((b_0, b_1, b_2, b_3, b_4)^T, \sigma^2\Sigma_0)\\
1/\sigma^2 \ ~\sim ~& \Ga(\nu_0/2, \nu_0\sigma_0^2/2) 
\end{aligned}
$$

This gives us the multivariate Normal-Gamma conjugate family, with hyperparameters $b_0, b_1, b_2, b_3, b_4, \Sigma_0, \nu_0$, and $\sigma_0^2$. For this prior, we will need to specify the values of all the hyperparameters. This elicitation can be quite involved, especially when we do not have enough prior information about the variances, covariances of the coefficients and other prior hyperparameters. Therefore, we are going to adopt the noninformative reference prior, which is a limiting case of this multivariate Normal-Gamma prior.

The reference prior in the multiple linear regression model is similar to the reference prior we used in the simple linear regression model. The prior distribution of all the coefficients $\beta$'s conditioning on $\sigma^2$ is the uniform prior, and the prior of $\sigma^2$ is proportional to its reciprocal
$$ p(\beta_0,\beta_1,\beta_2,\beta_3,\beta_4~|~\sigma^2) \propto 1,\qquad\quad p(\sigma^2) \propto \frac{1}{\sigma^2}. $$

Under this reference prior, the marginal posterior distributions of the coefficients, $\beta$'s, are parallel to the ones in simple linear regression. The marginal posterior distribution of $\beta_j$ is the Student's $t$-distributions with centers given by the frequentist OLS estimates  $\hat{\beta}_j$, scale parameter given by the standard error $(\text{se}_{\beta_j})^2$ obtained from the OLS estimates
$$
\beta_j~|~y_1,\cdots,y_n ~\sim ~\St(n-p-1,\ \hat{\beta}_j,\ (\text{se}_{\beta_j})^2),\qquad j = 0, 1, \cdots, p.
$$

The degree of freedom of these $t$-distributions is $n-p-1$, where $p$ is the number of predictor variables. In the kid's cognitive score example, $p=4$. The posterior mean, $\hat{\beta}_j$, is the center of the $t$-distribution of $\beta_j$, which is the same as the OLS estimates of $\beta_j$. The posterior standard deviation of $\beta_j$, which is the square root of the scale parameter of the $t$-distribution, is $\text{se}_{\beta_j}$, the standard error of $\beta_j$ under the OLS estimates. That means, under the reference prior, we can easily obtain the posterior mean and posterior standard deviation from using the `lm` function, since they are numerically equivalent to the counterpart of the frequentist approach.


### Fitting the Bayesian Model

To gain more flexibility in choosing priors, we will instead use the `bas.lm` function in the `BAS` library, which allows us to specify different model priors and coefficient priors.


```r
# Import library
library(BAS)

# Use `bas.lm` to run regression model
cog.bas = bas.lm(kid_score ~ ., data = cognitive, prior = "BIC", 
                 modelprior = Bernoulli(1), 
                 include.always = ~ ., 
                 n.models = 1)
```

The above `bas.lm` function uses the same model formula  as in the `lm`. It first specifies the response and predictor variables, a data argument to provide the data frame. The additional arguments further include the prior on the coefficients. We use `"BIC"` here to indicate that the model is based on the non-informative reference prior. (We will explain in the later section why we use the name `"BIC"`.) Since we will only provide one model, which is the one that includes all variables, we place all model prior probability to this exact model. This is specified in the `modelprior = Bernoulli(1)` argument. Because we want to fit using all variables, we use `include.always = ~ .` to indicate that the intercept and all 4 predictors are included. The argument `n.models = 1` fits just this one model.


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

From the last column in this summary, we see that the probability of the coefficients to be non-zero is always 1. This is because we specify the argument `include.always = ~ .` to force the model to include all variables. Notice on the first row we have the statistics of the `Intercept` $\beta_0$. The posterior mean of $\beta_0$ is 86.8, which is completely different from the original $y$-intercept of this model under the frequentist OLS regression. As we have stated previously, we consider the "centered" model under the Bayesian framework. Under this "centered" model and the reference prior, the posterior mean of the `Intercept` $\beta_0$ is now the sample mean of the response variable $Y_{\text{score}}$.

We can visualize the coefficients $\beta_1,\ \beta_2,\ \beta_3,\ \beta_4$ using the `plot` function. We use the `subset` argument to plot only the coefficients of the predictors.


```r
par(mfrow = c(2, 2), col.lab = "darkgrey", col.axis = "darkgrey", col = "darkgrey")
plot(cog.coef, subset = 2:5, ask = F)
```

![](06-regression-03-Bayesian-multi-regression_files/figure-latex/plot-coef-1.pdf)<!-- --> 

These distributions all center the posterior distributions at their respective OLS estimates $\hat{\beta}_j$, with the spread of the distribution related to the standard errors $\text{se}_{\beta_j}$.  Recall, that `bas.lm` uses centered predictors so that the intercept is always the sample mean.

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
out = confint(cog.coef)[, 1:2]  

# Extract the upper and lower bounds of the credible intervals
names = c("posterior mean", "posterior std", colnames(out))
out = cbind(cog.coef$postmean, cog.coef$postsd, out)
colnames(out) = names

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

As in the simple linear aggression, the posterior estimates from the reference prior, that are in the table, are **equivalent to the numbers** reported from the `lm` function in R, or using the confident function in the OLS estimates. These intervals are centered at the posterior mean $\hat{\beta}_j$ with width given by the appropriate $t$ quantile with $n-p-1$ degrees of freedom times the posterior standard deviation $\text{se}_{\beta_j}$. **The primary difference is the interpretation of the intervals**. For example, given this data, we believe there is a 95% chance that the kid's cognitive score increases by 0.44 to 0.68 with one additional increase of the mother's IQ score. The mother's high school status has a larger effect where we believe that there is a 95% chance the kid would score of 0.55 up to 9.64 points higher if the mother had three or more years of high school. The credible intervals of the predictors `work` and `age` include 0, which implies that we may improve this model so that the model will accomplish a desired level of explanation or prediction with fewer predictors. We will explore model selection using Bayesian information criterion in the next chapter.

## Summary

We have provided Bayesian analyses for both simple linear regression and multiple linear regression using the default reference prior. We have seen that, under this reference prior, the marginal posterior distribution of the coefficients is the Student's $t$-distribution. Therefore, the posterior mean and posterior standard deviation of any coefficients are numerically equivalent to the corresponding frequentist OLS estimate and the standard error. This has provided us a base line analysis of Bayesian approach, which we can extend later when we introduce more different coefficient priors. 

The difference is the interpretation. Since we have obtained the distribution of each coefficient, we can construct the credible interval, which provides us the probability that a specific coefficient falls into this credible interval. 

We have also used the posterior distribution to analyze the probability of a particular observation being an outlier. We defined such probabiilty to be the probability that the error term is $k$ standard deviations away from 0. This probability is based on information of all data, instead of just the observation itself. 
