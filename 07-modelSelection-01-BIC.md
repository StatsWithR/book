## Bayesian Information Criterion (BIC) {#sec:BIC}

In inferential statistics, we compare model selections using $p$-values or adjusted $R^2$. Here we will take the Bayesian propectives. We are going to discuss the Bayesian model selections using the Bayesian information criterion, or BIC. BIC is one of the Bayesian criteria used for Bayesian model selection, and tends to be one of the most popular criteria. 


### Definition of BIC

The Bayesian information criterion, BIC, is defined to be

\begin{equation}
\text{BIC} = -2\ln(\widehat{\text{likelihood}}) + (p+1)\ln(n).
(\#eq:BIC-def)
\end{equation}

Here $n$ is the number of observations in the model, and $p$ is the number of predictors. That is, $p+1$ is the number of total parameters (also the total number of coefficients, including the intercept) in the model. Recall that in the Bayesian simple linear regression (Section \@ref(sec:simple-linear)), we mentioned the likelihood of the model $y_i=\alpha + \beta x_i+\epsilon_i$ is the probability (or probability distribution) for the observed data $y_i,\  i = 1,\cdots, n$ occur under the given parameters $\alpha,\ \beta,\ \sigma^2$
$$ \text{likelihood} = p(y_i~|~\alpha,\ \beta, \ \sigma^2) = \mathcal{L}(\alpha,\ \beta,\ \sigma^2), $$
where $\sigma^2$ is the variance of the assumed Normal distribution of the error term $\epsilon_i$. In general, under any model $M$, we can write the likelihood of this model as the function of parameter $\boldsymbol{\theta}$ ($\boldsymbol{\theta}$ may be a vector of several parameters) and the model $M$
$$ \text{likelihood} = p(\text{data}~|~\boldsymbol{\theta}, M) = \mathcal{L}(\boldsymbol{\theta}, M). $$
If the likelihood function $\mathcal{L}(\boldsymbol{\theta}, M)$ is nice enough (say it has local maximum), the maximized value of the likelihood, $\widehat{\text{likelihood}}$, can be achieved by some special value of the parameter $\boldsymbol{\theta}$, denoted as $\hat{\boldsymbol{\theta}}$
$$ \widehat{\text{likelihood}} = p(\text{data}~|~\hat{\boldsymbol{\theta}}, M) = \mathcal{L}(\hat{\boldsymbol{\theta}}, M).$$

This is the likelihood that defines BIC.

When the sample size $n$ is large enough and the data distribution belongs to the exponential family such as the Normal distribution, BIC can be approximated by -2 times likelihood that data are produced under model $M$:

\begin{equation}
\text{BIC}\approx -2\ln(p(\text{data}~|~M)) = -2\ln\left(\int p(\text{data}~|~\boldsymbol{\theta}, M)p(\boldsymbol{\theta}~|~M)\, d\boldsymbol{\theta}\right),\qquad \quad \text{when $n$ is large.} (\#eq:BIC-approx)
\end{equation}

Here $p(\boldsymbol{\theta}~|~M)$ is the prior distribution of the parameter $\boldsymbol{\theta}$. We will not go into detail why the approximation holds and how we perform the integration in this book. However, we wanted to remind readers that, since BIC can be approximated by the prior distribution of the parameter $\boldsymbol{\theta}$, we will see later how we utilize BIC to approximate the model likelihood under the reference prior.^[Recall that the reference prior is the limiting case of the multivariate Normal-Gamma distribution.]

One more observation of formula \@ref(eq:BIC-approx) is that it involves the marginal likelihood of data under model $M$, $p(\text{data}~|~M)$. We have seen this quantity when we introduced Bayes factor between two hypotheses or models
$$ \BF[M_1:M_2] = \frac{p(\text{data}~|~M_1)}{p(\text{data}~|~M_2)}. $$
This also provides connection between BIC and Bayes factor, which we will leverage later.


Similar to AIC, the Akaike information criterion, the model with the smallest BIC is preferrable. Formula \@ref(eq:BIC-def) can be re-expressed using the model $R^2$, which is easier to calculate
\begin{equation}
\text{BIC} = n\ln(1-R^2)+(p+1)\ln(n)+\text{constant},
(\#eq:BIC-new)
\end{equation}
where the last term constant only depends on the sample size $n$, and the observed data $y_1,\cdots, y_n$. Since this constant does not depend on the choice of model, i.e., the choice of variables, ignoring this constant will not affect the comparison of BICs between models. Therefore, we usually define BIC to be
\begin{equation*}
\text{BIC} = n\ln(1-R^2) + (p+1)\ln(n).
\end{equation*}


From this expression, we see that adding more predictors, that is, increasing $p$, will result in larger $R^2$, which leads to a smaller $\ln(1-R^2)$ in the first term of BIC. While larger $R^2$ means better goodness of fit of the data, too many predictors may result in overfitting the data. Therefore, the second term $(p+1)\ln(n)$ is added in the BIC expression to penalize models with too many predictors. When $p$ increases, the second term increases as well. This provides a trade-off between the goodness of fit given by the first term and the model complexity represented by the second term.

### Backward Elimination with BIC

We will use the kid's cognitive score data set `cognitive` as an example. We first read in the data set from Gelman's website and transform the data types of the two variables `mom_work` and `mom_hs`, like what we did in Section \@ref(sec:Bayes-multiple-regression).


```r
# Load the library in order to read in data from website
library(foreign)    

# Read in cognitive score data set and process data tranformations
cognitive = read.dta("http://www.stat.columbia.edu/~gelman/arm/examples/child.iq/kidiq.dta")

cognitive$mom_work = as.numeric(cognitive$mom_work > 1)
cognitive$mom_hs =  as.numeric(cognitive$mom_hs > 0)
colnames(cognitive) = c("kid_score", "hs","IQ", "work", "age")
```

We start with the full model, with all possible predictors: `hs`, `IQ`, `work`, and `age`. We will drop one variable at a time and record all BICs. Then we will choose the model with the smallest BIC. We will repeat this process until none of the models yields a decrease in BIC. We use the `step` function in R to perform the BIC model selection. Notice the default value of the `k` argument in the `step` function is `k=2`, which is for the AIC score. For BIC, `k` should be `log(n)` correspondingly.


```r
# Compute the total number of observations
n = nrow(cognitive)

# Full model using all predictors
cog.lm = lm(kid_score ~ ., data=cognitive)

# Perform BIC elimination from full model
# k = log(n): penalty for BIC rather than AIC
cog.step = step(cog.lm, k=log(n))   
```

```
## Start:  AIC=2541.07
## kid_score ~ hs + IQ + work + age
## 
##        Df Sum of Sq    RSS    AIC
## - age   1     143.0 141365 2535.4
## - work  1     383.5 141605 2536.2
## - hs    1    1595.1 142817 2539.9
## <none>              141222 2541.1
## - IQ    1   28219.9 169441 2614.1
## 
## Step:  AIC=2535.44
## kid_score ~ hs + IQ + work
## 
##        Df Sum of Sq    RSS    AIC
## - work  1     392.5 141757 2530.6
## - hs    1    1845.7 143210 2535.0
## <none>              141365 2535.4
## - IQ    1   28381.9 169747 2608.8
## 
## Step:  AIC=2530.57
## kid_score ~ hs + IQ
## 
##        Df Sum of Sq    RSS    AIC
## <none>              141757 2530.6
## - hs    1    2380.2 144137 2531.7
## - IQ    1   28504.1 170261 2604.0
```

In the summary chart, the `AIC` should be interpreted as BIC, since we have chosen to use the BIC expression where $k=\ln(n)$.

From the full model, we predict the kid's cognitive score from mother's high school status, mother's IQ score, mother's work status and mother's age. The BIC for the full model is 2541.1.

At the first step, we try to remove each variable from the full model to record the resulting new BIC. From the summary statistics, we see that removing variable `age` results in the smallest BIC. But if we try to drop the `IQ` variable, this will increase the BIC, which implies that `IQ` would be a really important predictor of `kid_score`. Comparing all the results, we drop the `age` variable at the first step. After dropping `age`, the new BIC is 2535.4.

At the next step, we see that dropping `work` variable will result in the lowest BIC, which is 2530.6. Now the model has become
$$ \text{score} \sim \text{hs} + \text{IQ} $$

Finally, when we try dropping either `hs` or `IQ`, it will result in higher BIC than 2530.6. This suggests that we have reached the best model. This model predicts kid's cognitive score using mother's high school status and mother's IQ score. 

However, using the adjusted $R^2$, the best model would be the one including not only `hs` and `IQ` variables, but also mother's work status, `work`. In general, using BIC leads to fewer variables for the best model compared to using adjusted $R^2$ or AIC.

We can also use the `BAS` package to find the best BIC model without taking the stepwise backward process.

```r
# Import library
library(BAS)

# Use `bas.lm` to run regression model
cog.BIC = bas.lm(kid_score ~ ., data = cognitive,
                 prior = "BIC", modelprior = uniform())

cog.BIC
```

```
## 
## Call:
## bas.lm(formula = kid_score ~ ., data = cognitive, prior = "BIC", 
##     modelprior = uniform())
## 
## 
##  Marginal Posterior Inclusion Probabilities: 
## Intercept         hs         IQ       work        age  
##   1.00000    0.61064    1.00000    0.11210    0.06898
```
Here we set the `modelprior` argument as `uniform()` to assign equal prior probability for each possible model.

The `logmarg` information inside the `cog.BIC` summary list records the log of marginal likelihood of each model after seeing the data $\ln(p(\text{data}~|~M))$. Recall that this is approximately proportional to negative BIC when the sample size $n$ is large
$$ \text{BIC}\approx -2 \ln(p(\text{data}~|~M)).$$

We can use this information to retreat the model with the largest log of marginal likelihood, which corresponds to the model with the smallest BIC. 


```r
# Find the index of the model with the largest logmarg
best = which.max(cog.BIC$logmarg)

# Retreat the index of variables in the best model, with 0 as the index of the intercept
bestmodel = cog.BIC$which[[best]]
bestmodel
```

```
## [1] 0 1 2
```

```r
# Create an indicator vector indicating which variables are used in the best model
bestgamma = rep(0, cog.BIC$n.vars) 

# Create a 0 vector with the same dimension of the number of variables in the full model
bestgamma[bestmodel + 1] = 1  

# Change the indicator to 1 where variables are used
bestgamma
```

```
## [1] 1 1 1 0 0
```

From the indicator vector `bestgamma` we see that only the intercept (indexed as 0), mother's high school status variable `hs` (indexed as 1), and mother's IQ score `IQ` (indexed as 2) are used in the best model, with 1's in the corresponding slots of the 5-dimensional vector $(1, 1, 1, 0, 0)$.


### Coefficient Estimates Under Reference Prior for Best BIC Model

The best BIC model $M$ can be set up as follows and we have adopted the "centered" model convention for convenient analyses
$$ y_{\text{score},i} = \beta_0 + \beta_1(x_{\text{hs},i} - \bar{x}_{\text{hs}, i})+\beta_2(x_{\text{IQ},i}-\bar{x}_{\text{IQ}})+\epsilon_i,\qquad \quad i = 1,\cdots, n $$

We would like to get the posterior distributions of the coefficients $\beta_0$, $\beta_1$, and $\beta_2$ under this model. Recall that the reference prior imposes a uniformly flat prior distribution on coefficients $p(\beta_0, \beta_1, \beta_2~|~M)\propto 1$ and that $p(\sigma^2~|~M) \propto 1/\sigma^2$, so together the joint prior distribution $p(\beta_0, \beta_1, \beta_2, \sigma^2~|~M)$ is proportional to $1/\sigma^2$. When the sample size $n$ is large, any proper prior distribution $p(\beta_0, \beta_1, \beta_2, \sigma^2~|~M)$ is getting flatter and flatter, which can be approximated by the reference prior. At the same time, the log of marginal likelihood $\ln(p(\text{data}~|~M))$ can be approximated by the BIC. Therefore, we use `prior = "BIC"` in the `bas.lm` function when we use the BIC as an approximation of the log of marginal likelihood under the reference prior. The posterior mean of $\beta_0$ in the result is the sample mean of the kids' cognitive scores, or $\bar{Y}_{\text{score}}$, since we have centered the model. 


```r
# Fit the best BIC model by imposing which variables to be used using the indicators
cog.bestBIC = bas.lm(kid_score ~ ., data = cognitive,
                     prior = "BIC", n.models = 1,  # We only fit 1 model
                     bestmodel = bestgamma,  # We use bestgamma to indicate variables 
                     modelprior = uniform())
```

```
## Warning in model.matrix.default(mt, mf, contrasts): non-list contrasts
## argument ignored
```

```r
# Retreat coefficients information
cog.coef = coef(cog.bestBIC)

# Retreat bounds of credible intervals
out = confint(cog.coef)[, 1:2]

# Combine results and construct summary table
coef.BIC = cbind(cog.coef$postmean, cog.coef$postsd, out)
names = c("post mean", "post sd", colnames(out))
colnames(coef.BIC) = names
coef.BIC
```

```
##           post mean    post sd       2.5%      97.5%
## Intercept 86.797235 0.87054033 85.0862025 88.5082675
## hs         5.950117 2.21181218  1.6028370 10.2973969
## IQ         0.563906 0.06057408  0.4448487  0.6829634
## work       0.000000 0.00000000  0.0000000  0.0000000
## age        0.000000 0.00000000  0.0000000  0.0000000
```

Comparing the coefficients in the best model with the ones in the full model (which can be found in Section \@ref(sec:Bayes-multiple-regression)), we see that the 95\% credible interval for `IQ` variable is the same. However, the credible interval for high school status `hs` has shifted slightly to the right, and it is also slighly narrower, meaning a smaller posterior standard deviation. All credible intervals of coefficients exclude 0, suggesting that we have found a parsimonious model.^[A parsimonious model is a model that accomplishes a desired level of explanation or prediction with as few predictor variables as possible. More discussion of parsimonious models can be found in Course 3 Linear Regression and Modeling.]

### Other Criteria

BIC is one of the criteria based on penalized likelihoods. Other examples such as AIC (Akaike information criterion) or adjusted $R^2$, employ the form of 
$$ -2\ln(\widehat{\text{likelihood}}) + (p+1)\times\text{some constant},$$
where $p$ is the number of predictor variables and "some constant" is a constant value depending on different criteria. BIC tends to select parsimonious models (with fewer predictor variables) while AIC and adjusted $R^2$ may include variables that are not statistically significant, but may do better for predictions.

Other Bayesian model selection decisions may be based on selecting models with the highest posterior probability. If predictions are important, we can use decision theory to help pick the model with the smallest expected prediction error. In addiciton to goodness of fit and parsimony, loss functions that include costs associated with collecting variables for predictive models may be of important consideration. 
