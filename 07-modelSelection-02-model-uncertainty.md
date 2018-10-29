## Bayesian Model Uncertainty {#sec:BMU}

In the last section, we discussed how to use Bayesian Information Criterion (BIC) to pick the best model, and we demonstrated the method on the kid's cognitive score data set. However, we may often  have several models with similar BIC. If we only pick the one with the lowest BIC, we may ignore the presence of other models that are equally good or can provide useful information. The credible intervals of coefficients may be narrower since the uncertainty is being ignored when we consider only one model. Narrower intervals are not always better if they miss the true values of the parameters. To account for the uncertainty, getting the posterior probability of all possible models is necessary. In this section, we will talk about how to convert BIC into Bayes factor to find the posterior probability of all possible models. We will again use the `BAS` package in R to achieve this goal.

### Model Uncertainty

When forecasting the path of a hurricane, having an accurate prediction and measurement of uncertainty is important for early warning. In this case, we would consider the probability of several potential paths that the hurricane may make landfall. Similar to hurricane forecasting, we would also like to obtain the posterior probability of all possible models for uncertainty measurement. 

To represent model uncertainty, we need to construct a probability distribution over all possible models where the each probability provides measure of how likely the model is to happen.

Suppose we have a multiple linear regression
$$ y_i = \beta_0+\beta_1(x_{1,i} - \bar{x}_1)+\beta_2(x_{2,i} - \bar{x}_2)+\cdots+\beta_p(x_{p,i}-\bar{x}_p)+\epsilon_i, \quad 1\leq i \leq n,$$
with $p$ predictor variables $x_1,\cdots, x_p$. There are in total $2^p$ different models, corresponding to $2^p$ combinations of variable selections. there are 2 possibilities for each variable: either getting selected or not, and we have in total $p$ variables. We denote each model as $M_m,\ m=1,\cdots,2^p$. To obtian the posterior probability of each model $p(M_m~|~\text{data})$, Bayes' rule tells that that we need to assign the prior probability $p(M_m)$ to each model, and to then obtain the marginal likelihood of each model $p(\text{data}~|~M_m)$. By Bayes' rule, we update the posterior probability of each model $M_m$ after seeing the date, via marginal likelihood of model $M_m$:

\begin{equation} 
p(M_m~|~\text{data}) = \frac{\text{marginal likelihood of }M_m\times p(M_m)}{\sum_{j=1}^{2^p}\text{marginal likelihood of }M_j\times p(M_j)} = \frac{p(\text{data}~|~M_m)p(M_m)}{\sum_{j=1}^{2^p}p(\text{data}~|~M_m)p(M_m)}. 
(\#eq:model-post-prob) 
\end{equation}

The marginal likelihood $p(\text{data}~|~M_m)$ of each model $M_m$ serves to reweight the prior probability $p(M_m)$, so that models with higher likelihoods have larger weights, and models with lower likelihoods receive smaller weights. We renormalize this weighted prior probability by dividing it by the sum $\displaystyle \sum_{j=1}^{2^p}p(\text{data}~|~M_j)p(M_j)$ to get the posterior probability of each model.

Recall that the prior odd between two models $M_1$ and $M_2$ is defined to be

$$
\Odd[M_1:M_2] =  \frac{p(M_1)}{p(M_2)},
$$
and the Bayes factor is defined to be the ratio of the likelihoods of two models
$$ \BF[M_1:M_2] = \frac{p(\text{data}~|~M_1)}{p(\text{data}~|~M_2)}. $$

Suppose we have chosen a base model $M_b$, we may divide both the numerator and the denominator of the formula \@ref(eq:model-post-prob) by $p(\text{data}~|~M_b)\times p(M_b)$. This gives us a new formula to calculate the posterior probability of model $M_m$ based on the prior odd and the Bayes factor. In this new formula, we can see that the evidence from the data in the Bayes factor $\BF[M_j:M_b],\ j=1,\cdots, 2^p$ serve to upweight or downweight the prior odd $\Odd[M_j:M_b],\ j=1,\cdots,2^p$.

$$
\begin{aligned}
p(M_m~|~\text{data}) = & \frac{p(\text{data}~|~M_m)\times p(M_m)/(p(\text{data}~|~M_b)\times p(M_b))}{\sum_{j=1}^{2^p}(p(\text{data}~|~M_j)\times p(M_j))/(p(\text{data}~|~M_b)\times p(M_b))} \\
 & \\
= & \frac{[p(\text{data}~|~M_m)/p(\text{data}~|~M_b)]\times[p(M_m)/p(M_b)]}{\sum_{j=1}^{2^p}[p(\text{data}~|~M_j)/p(\text{data}~|~M_b)]\times[p(M_j)/p(M_b)]}\\
  & \\
= & \frac{\BF[M_m:M_b]\times \Odd[M_m:M_b]}{\sum_{j=1}^{2^p}\BF[M_j:M_b]\times \Odd[M_j:M_b]}.
\end{aligned}
$$
Any model can be used as the base model $M_b$. It could be the model with the highest posterior probability, or the null model $M_0$ with just the intercept $y_i = \beta_0+\epsilon_i$. 

Using BIC, we can approximate the Bayes factor between two models by their OLS $R$-squared's and the numbers of predictors used in the models, when we have large sample of data. This provides a much easier way to approximate the posterior probability of models since obtaining $R^2$ can be done by the usual OLS linear regression. Recall that in Section \@ref(sec:BIC), we provided the fact that BIC of any model $M_m$ (denoted as $\text{BIC}_m$) is an asymptotic approximation of the log of marginal likelihood of $M_m$ when the sample size $n$ is large (Equation \@ref(eq:BIC-approx))
$$ \text{BIC}_m \approx -2 \ln(\text{marginal likelihood}) = -2\ln(p(\text{data}~|~M_m)). $$

Using this fact, we can approximate Bayes factor between two models by their BICs
$$ \BF[M_1:M_2] = \frac{p(\text{data}~|~M_1)}{p(\text{data}~|~M_2)} \approx \frac{\exp(-\text{BIC}_1/2)}{\exp(-\text{BIC}_2/2)}=\exp\left(-\frac{1}{2}(\text{BIC}_1-\text{BIC}_2)\right).$$

We also know that BIC can be calculated by the OLS $R^2$ and the number of predictors $p$ from Equation \@ref(eq:BIC-new) in Section \@ref(sec:BIC)
$$ \text{BIC} = n\ln(1-R^2) + (p+1)\ln(n) + \text{constant}. $$
(We usually ignore the constant in the last term since it does not affect the difference betweeen two BICs.)

Using this formula, we can approximate Bayes factor between model $M_1$ and $M_2$ by their corresponding $R$-squared's and the numbers of predictors
\begin{equation}
\BF[M_1:M_2]\approx \left(\frac{1-R_1^2}{1-R_2^2}\right)^{\frac{n}{2}}\times n^{\frac{p_1-p_2}{2}}.
(\#eq:BF-Rsquared)
\end{equation}

As for the null model $M_0:\ y_i = \beta_0+\epsilon_i$, $R_0^2 = 0$ and $p_0=0$. Equation \@ref(eq:BF-Rsquared) can be further simplified as 
$$ \BF[M_m:M_0] = (1-R_m^2)^{\frac{n}{2}}\times n^{\frac{p_m}{2}}. $$


### Calculating Posterior Probability in R

Back to the kid's cognitive score example, we will see how the summary of results using `bas.lm` tells us about the posterior probability of all possible models.




Suppose we have already loaded the data and pre-processed the columns `mom_work` and `mom_hs` using `as.numeric` function, as what we did in the last section. To represent model certainty, we construct the probability distribution overall possible 16 (=$2^4$) models where each probability $p(M_m)$ provides a measure of how likely the model $M_m$ is. Inside the `bas.lm` function, we first specify the full model, which in this case is the `kid_score`, being regressed by all predictors: mother's high school status `hs`, mother's IQ `IQ`, mother's work status `work` and mother's age `age`. We take the `data = cognitive` in the next argument. For the prior distribution of the coefficients for calculating marginal likelihoods, we use `prior = "BIC"` to approximate the marginal likelihood $p(\text{data}~|~M_m)$. We then use `modelprior = uniform()` in the argument to assign equal prior probability $p(M_m),\ m=1,\cdots, 16$ to all 16 models. That is, $\displaystyle p(M_m) = \frac{1}{16}$.


```r
# Import libary
library(BAS)

# Use `bas.lm` for regression
cog_bas = bas.lm(kid_score ~ hs + IQ + work + age, 
                 data = cognitive, prior = "BIC",
                 modelprior = uniform())
```

`cog_bas` is a `bas` object. The usual `print`, `summary`, `plot`, `coef`, `fitted`, `predict` functions are available and can be used on `bas` objects similar to `lm` objects created by the usual `lm` function. From calling


```r
names(cog_bas)
```

```
##  [1] "probne0"        "which"          "logmarg"        "postprobs"     
##  [5] "priorprobs"     "sampleprobs"    "mse"            "mle"           
##  [9] "mle.se"         "shrinkage"      "size"           "R2"            
## [13] "rank"           "rank_deficient" "n.models"       "namesx"        
## [17] "n"              "prior"          "modelprior"     "alpha"         
## [21] "probne0.RN"     "postprobs.RN"   "include.always" "df"            
## [25] "n.vars"         "Y"              "X"              "mean.x"        
## [29] "call"           "xlevels"        "terms"          "model"
```
one can see the outputs and analyses that we can extract from a `bas` object.

The `bas` object takes the `summary` method


```r
round(summary(cog_bas), 3)
```

```
##           P(B != 0 | Y)   model 1   model 2   model 3   model 4   model 5
## Intercept         1.000     1.000     1.000     1.000     1.000     1.000
## hs                0.611     1.000     0.000     0.000     1.000     1.000
## IQ                1.000     1.000     1.000     1.000     1.000     1.000
## work              0.112     0.000     0.000     1.000     1.000     0.000
## age               0.069     0.000     0.000     0.000     0.000     1.000
## BF                   NA     1.000     0.562     0.109     0.088     0.061
## PostProbs            NA     0.529     0.297     0.058     0.046     0.032
## R2                   NA     0.214     0.201     0.206     0.216     0.215
## dim                  NA     3.000     2.000     3.000     4.000     4.000
## logmarg              NA -2583.135 -2583.712 -2585.349 -2585.570 -2585.939
```

The summary table shows us the following information of the top 5 models

Item          | Description
--------------| -------------------------------------------------
`P(B!=0 | Y)` | Posterior inclusion probability (pip) of each coefficient under data $Y$
`0` or `1` in the column | indicator of whether the variable is included in the model
`BF`          | Bayes factor $\BF[M_m:M_b]$, where $M_b$ is the model with highest posterior probability
`PostProbs`   | Posterior probability of each model
`R2`          | $R$-squared in the ordinary least square (OLS) regression
`dim`         | Number of variables (including the intercept) included in the model
`logmarg`     | Log of marginal likelihood of the model, which is approximately  $-\displaystyle\frac{1}{2}\text{BIC}$

<br/>



All top 5 models suggest to exclude `age` variable and include `IQ` variable. The first model includes intercept $\beta_0$ and only `hs` and `IQ`, with a posterior probability of about 0. The model with the 2nd highest posterior probability, which includes only the intercept and the variable `IQ`, has posterior probability of about 0. These two models compose of total posterior probability of about 0, leaving only 1 posterior probability to the remaining 14 models. 

Using the `print` method, we obtain the marginal posterior inclusion probability (pip) $p(\beta_j\neq 0)$ of each variable $x_j$.

<!--

-->



```r
print(cog_bas)
```

```
## 
## Call:
## bas.lm(formula = kid_score ~ hs + IQ + work + age, data = cognitive, 
##     prior = "BIC", modelprior = uniform())
## 
## 
##  Marginal Posterior Inclusion Probabilities: 
## Intercept         hs         IQ       work        age  
##   1.00000    0.61064    1.00000    0.11210    0.06898
```



