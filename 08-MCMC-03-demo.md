## R Demo on `BAS` Package

In this section, we will apply Bayesian model selection and model averaging on the US crime data set `UScrime` using the `BAS` package. We will introduce some additional diagnostic plots, and talk about the effect of multicollinearity in model uncertainty.

### The `UScrime` Data Set and Data Processing



We will demo the `BAS` commands using the US crime data set in the R libarry `MASS`.


```r
# Load library and data set
library(MASS)
data(UScrime)
```


This data set contains data on 47 states of the US for the year of 1960. The response variable $Y$ is the rate of crimes in a particular category per head of population of each state. There are 15 potential explanatory variables with values for each of the 47 states related to crime and other demographics. Here is the table of all the potential explanatory variables and their descriptions.

Variable | Description
---------|---------------------------------------------
`M`      | Percentage of males aged 14-24
`So`     | Indicator variable for southern states
`Ed`     | Mean years of schooling
`Po1`    | Police expenditure in 1960
`Po2`    | Police expenditure in 1959
`LF`     | Labour force participation rate
`M.F`    | Number of males per 1000 females
`Pop`    | State population
`NW`     | Number of non-whites per 1000 people
`U1`     | Unemployment rate of urban males aged 14-24
`U2`     | Unemployment rate of urban males aged 35-39
`GDP`    | Gross domestic product per head
`Ineq`   | Income inequality
`Prob`   | Probability of imprisonment
`Time`   | Average time served in state prisons

<br/>

We may use the `summary` function to describe each variable in the data set. 

```r
summary(UScrime)
```

```
##        M               So               Ed             Po1       
##  Min.   :119.0   Min.   :0.0000   Min.   : 87.0   Min.   : 45.0  
##  1st Qu.:130.0   1st Qu.:0.0000   1st Qu.: 97.5   1st Qu.: 62.5  
##  Median :136.0   Median :0.0000   Median :108.0   Median : 78.0  
##  Mean   :138.6   Mean   :0.3404   Mean   :105.6   Mean   : 85.0  
##  3rd Qu.:146.0   3rd Qu.:1.0000   3rd Qu.:114.5   3rd Qu.:104.5  
##  Max.   :177.0   Max.   :1.0000   Max.   :122.0   Max.   :166.0  
##       Po2               LF             M.F              Pop        
##  Min.   : 41.00   Min.   :480.0   Min.   : 934.0   Min.   :  3.00  
##  1st Qu.: 58.50   1st Qu.:530.5   1st Qu.: 964.5   1st Qu.: 10.00  
##  Median : 73.00   Median :560.0   Median : 977.0   Median : 25.00  
##  Mean   : 80.23   Mean   :561.2   Mean   : 983.0   Mean   : 36.62  
##  3rd Qu.: 97.00   3rd Qu.:593.0   3rd Qu.: 992.0   3rd Qu.: 41.50  
##  Max.   :157.00   Max.   :641.0   Max.   :1071.0   Max.   :168.00  
##        NW              U1               U2             GDP       
##  Min.   :  2.0   Min.   : 70.00   Min.   :20.00   Min.   :288.0  
##  1st Qu.: 24.0   1st Qu.: 80.50   1st Qu.:27.50   1st Qu.:459.5  
##  Median : 76.0   Median : 92.00   Median :34.00   Median :537.0  
##  Mean   :101.1   Mean   : 95.47   Mean   :33.98   Mean   :525.4  
##  3rd Qu.:132.5   3rd Qu.:104.00   3rd Qu.:38.50   3rd Qu.:591.5  
##  Max.   :423.0   Max.   :142.00   Max.   :58.00   Max.   :689.0  
##       Ineq            Prob              Time             y         
##  Min.   :126.0   Min.   :0.00690   Min.   :12.20   Min.   : 342.0  
##  1st Qu.:165.5   1st Qu.:0.03270   1st Qu.:21.60   1st Qu.: 658.5  
##  Median :176.0   Median :0.04210   Median :25.80   Median : 831.0  
##  Mean   :194.0   Mean   :0.04709   Mean   :26.60   Mean   : 905.1  
##  3rd Qu.:227.5   3rd Qu.:0.05445   3rd Qu.:30.45   3rd Qu.:1057.5  
##  Max.   :276.0   Max.   :0.11980   Max.   :44.00   Max.   :1993.0
```

However, these variables have been pre-processed for modeling purpose, so the summary statistics may not be so meaningful. The values of all these variables have been aggregated over each state, so this is a case of ecological regression. We will not model directly the rate for a person to commit a crime. Instead, we will use the total number of crimes and average values of predictors at the state level to predict the total crime rate of each state.

We transform the variables using the natural log function, except the indicator variable `So` (2nd column of the data set). We perform this transformation based on the analysis of this data set.^[More details can be found in @venables2013modern.] Notice that `So` is already a numeric variable (`1` indicating Southern state and `0` otherwise), not as a categorical variable. Hence we do not need any data processing of this variable, unlike mother's high school status `hs` and mother's work status `work` in the kid's cognitive score data set.


```r
UScrime[,-2] = log(UScrime[,-2])
```


### Bayesian Models and Diagnostics

We run `bas.lm` function from the `BAS` package. We first run the full model and use this information for later decision on what variables to include. Here we have 15 potential predictors. The total number of models is$\ 2^{15} = 32768$. This is not a very large number and `BAS` can enumerate all the models pretty quickly. However, we want to illustrate how to explore models using stochastic methods. Hence we set argument `method = MCMC` inside the `bas.lm` function. We also use the Zellner-Siow cauchy prior for the prior distributions of the coefficients in this regression.


```r
library(BAS)
crime.ZS =  bas.lm(y ~ ., data=UScrime,
                   prior="ZS-null", modelprior=uniform(), method = "MCMC") 
```

`BAS` will run the MCMC sampler until the number of unique models in the sample exceeds $\text{number of models} = 2^{p}$ (when $p < 19$) or until the number of MCMC iterations exceeds $2\times\text{number of models}$ by default, whichever is smaller. Here $p$ is the number of predictors. 

**Diagnostic Plots**


To analyze the result, we first look at the diagnostic plot using `diagnostics` function and see whether we have run the MCMC exploration long enough so that the posterior inclusion probability (pip) has converged.


```r
diagnostics(crime.ZS, type="pip", col = "blue", pch = 16, cex = 1.5)
```



\begin{center}\includegraphics[width=0.8\linewidth]{08-MCMC-03-demo_files/figure-latex/diagnostics-1} \end{center}

In this plot, the $x$-axis is the renormalized posterior inclusion probability (pip) of each coefficient $\beta_i,\ i=1,\cdots, 15$ in this model. This can be calculated as
\begin{equation} 
p(\beta_i\neq 0~|~\text{data}) = \sum_{M_m\in\text{ model space}}I(X_i\in M_m)\left(\frac{\BF[M_m:M_0]\Odd[M_m:M_0]}{\displaystyle \sum_{M_j}\BF[M_j:M_0]\Odd[M_j:M_0]}\right).
(\#eq:pip)
\end{equation}

Here, $X_i$ is the $i$th predictor variable, and $I(X_i\in M_m)$ is the indicator function which is 1 if $X_i$ is included in model $M_m$ and 0 if $X_i$ is not included. The first $\Sigma$ notation indicates that we sum over all models $M_m$ in the model space. And we use
\begin{equation} 
\frac{\BF[M_m:M_0]\Odd[M_m:M_0]}{\displaystyle \sum_{M_j}\BF[M_j:M_0]\Odd[M_j:M_0]} 
(\#eq:weights)
\end{equation}
as the weights. You may recognize that the numerator of \@ref(eq:weights) is exactly the ratio of the posterior probability of model $M_m$ over the posterior probability of the null model $M_0$, i.e., the posterier odd $\PO[M_m:M_0]$. We devide the posterior odd by the total sum of posterior odds of all models in the model space, to make sure these weights are between 0 and 1. The weight in Equation \@ref(eq:weights) represents the posterior probability of the model $M_m$ after seeing the data $p(M_m~|~\text{data})$, the one we used in Section \@ref(sec:BMU). So Equation \@ref(eq:pip) is the theoretical calculation of pip, which can be rewrited as
$$ p(\beta_i\neq 0~|~\text{data}) = \sum_{M_m\in \text{ model space}}I(X_i\in M_m)p(M_m~|~\text{data}). $$
The null model $M_0$, as we recall, is the model that only includes the intercept.

On the $y$-axis of the plot, we lay out the posterior inclusion probability of coefficient $\beta_i$, which is calculated using
$$ p(\beta_i\neq 0~|~\text{data}) = \frac{1}{J}\sum_{j=1}^J I(X_i\in M^{(j)}).$$
Here $J$ is the total number of models that we sample using MCMC; each model is denoted as $M^{(j)}$ (some models may repeat themselves in the sample). We count the frequency of variable $X_i$ occuring in model $M^{(j)}$, and divide this number by the total number of models $J$. This is a frequentist approach to approximate the posterior probability of including $X_i$ after seeing the data.

When all points are on the 45 degree diagonal, we say that the posterior inclusion probability of each variable from MCMC have converged well enough to the theoretical posterior inclusion probability.

We can also use `diagnostics` function to see whether the model posterior probability has converged:

```r
diagnostics(crime.ZS, type = "model", col = "blue", pch = 16, cex = 1.5)
```



\begin{center}\includegraphics[width=0.7\linewidth]{08-MCMC-03-demo_files/figure-latex/model-prob-1} \end{center}

We can see that some of the points still fall slightly away from the 45 degree diagonal line. This may suggest we should increase the number of MCMC iterations. We may do that by imposing the argument on `MCMC.iterations` inside the `bas.lm` function


```r
# Re-run regression using larger number of MCMC iterations
crime.ZS = bas.lm(y ~ ., data = UScrime,
                  prior = "ZS-null", modelprior = uniform(),
                  method = "MCMC", MCMC.iterations = 10 ^ 6)

# Plot diagnostics again
diagnostics(crime.ZS, type = "model", col = "blue", pch = 16, cex = 1.5)
```



\begin{center}\includegraphics[width=0.8\linewidth]{08-MCMC-03-demo_files/figure-latex/more-MCMC-1} \end{center}

With more number of iterations, we see that most points stay in the 45 degree diagonal line, meaing the posterior inclusion probability from the MCMC method has mostly converged to the theoretical posterior inclusion probability.

We will next look at four other plots of the `BAS` object, `crime.ZS`. 

**Residuals Versus Fitted Values Using BMA**

The first plot is the residuals over the fitted value under Bayesian model averaging results.


```r
plot(crime.ZS, which = 1, add.smooth = F, 
     ask = F, pch = 16, sub.caption="", caption="")
abline(a = 0, b = 0, col = "darkgrey", lwd = 2)
```



\begin{center}\includegraphics[width=0.7\linewidth]{08-MCMC-03-demo_files/figure-latex/plot1-1} \end{center}

We can see that the residuals lie around the dash line $y=0$, and has a constant variance. Observations 11, 22, and 46 may be the potential outliers, which are indicated in the plot.

**Cumulative Sampled Probability**

The second plot shows the cumulative sampled model probability.


```r
plot(crime.ZS, which=2, add.smooth = F, sub.caption="", caption="")
```



\begin{center}\includegraphics[width=0.7\linewidth]{08-MCMC-03-demo_files/figure-latex/CMP-1} \end{center}

We can see that after we have discovered about 5,000 unique models with MCMC sampling, the probability is starting to level off, indicating that these additional models have very small probability and do not contribute substantially to the posterior distribution. These probabilities are proportional to the product of marginal likelihoods of models and priors, $p(\text{data}~|~M_m)p(M_m)$,  rather than Monte Carlo frequencies.

**Model Complexity**

The third plot is the model size versus the natural log of the marginal likelihood, or the Bayes factor, to compare each model to the null model.


```r
plot(crime.ZS, which=3, ask=F, caption="", sub.caption="")
```



\begin{center}\includegraphics[width=0.7\linewidth]{08-MCMC-03-demo_files/figure-latex/model-comp-1} \end{center}


We see that the models with the highest Bayes factors or logs of marginal likelihoods have around 8 or 9 predictors. The null model has a log of marginal likelihood of 0, or a Bayes factor of 1.

**Marginal Inclusion Probability**

Finally, we have a plot showing the importance of different predictors. 


```r
plot(crime.ZS, which = 4, ask = F, caption = "", sub.caption = "", 
     col.in = "blue", col.ex = "darkgrey", lwd = 3)
```



\begin{center}\includegraphics[width=0.7\linewidth]{08-MCMC-03-demo_files/figure-latex/MIP-1} \end{center}

The lines in blue correspond to the variables where the marginal posterior inclusion probability (pip), is greater than 0.5, suggesting that these variables are important for prediction. The variables represented in grey lines have posterior inclusion probability less than 0.5. Small posterior inclusion probability may arise when two or more variables are highly correlated, similar to large $p$-values with multicollinearity. So we should be cautious to use these posterior inclusion probabilities to eliminate variables.

**Model Space Visualization**

To focus on the high posterior probability models, we can look at the image of the model space.


```r
image(crime.ZS, rotate = F)
```

![](08-MCMC-03-demo_files/figure-latex/model-space-1.pdf)<!-- --> 

By default, we only include the top 20 models. An interesting feature of this plot is, that whenever `Po1`, the police expenditures in 1960, is included, `Po2`, the police expenditures in 1959, will be excluded from the model, and vice versa. 


```r
out = cor(UScrime$Po1, UScrime$Po2)
out 
```

```
## [1] 0.9933688
```

Calculating the correlation between the two variables, we see that that `Po1` and `Po2` are highly correlated with positive correlation 0.993.

### Posterior Uncertainty in Coefficients

Due to the interesting inclusion relationship between `Po1` and `Po2` in the top 20 models, we extract the two coefficients under Bayesian model averaging and take a look at the plots for the coefficients for `Po1` and `Po2`.


```r
# Extract coefficients
coef.ZS=coef(crime.ZS)

# Po1 and Po2 are in the 5th and 6th columns in UScrime
par(mfrow = c(1,2))
plot(coef.ZS, subset = c(5:6), 
     col.lab = "darkgrey", col.axis = "darkgrey", col = "darkgrey", ask = F)
```

![](08-MCMC-03-demo_files/figure-latex/Po1-Po2-plots-1.pdf)<!-- --> 

Under Bayesian model averaging, there is more mass at 0 for `Po2` than `Po1`, giving more posterior inclusion probability for `Po1`. This is also the reason why in the marginal posterior plot of variable importance, `Po1` has a blue line while `Po2` has a grey line. When `Po1` is excluded, the distributions of other coefficients in the model, except the one for `Po2`, will have similar distributions as when both `Po1` and `Po2` are in the model. However, when both predictors are included, the adjusted coefficient for `Po2` has more support on negative values, since we are over compensating for having both variables included in the model. In extreme cases of correlations, one may find that the coefficient plot is multimodal. If this is the case, the posterior mean may not be in the highest probability density credible interval, and this mean is not necessarily an informative summary. We will discuss more in the next section about making decisions on highly correlated variables.

We can read the credible intervals of each variable using the `confint` function on the coefficient object `coef.ZS` of the model. Here we round the results in 4 decimal places.


```r
round(confint(coef.ZS), 4)
```

```
##              2.5%  97.5%    beta
## Intercept  6.6662 6.7816  6.7249
## M          0.0000 2.1841  1.1493
## So        -0.0519 0.3061  0.0355
## Ed         0.6228 3.1788  1.8630
## Po1        0.0000 1.4379  0.6011
## Po2       -0.1555 1.4471  0.3188
## LF        -0.4884 1.0278  0.0602
## M.F       -2.3845 1.8954 -0.0311
## Pop       -0.1267 0.0053 -0.0224
## NW         0.0000 0.1668  0.0663
## U1        -0.5431 0.3387 -0.0248
## U2         0.0000 0.6704  0.2089
## GDP       -0.0779 1.1442  0.2063
## Ineq       0.6560 2.1301  1.3913
## Prob      -0.4135 0.0000 -0.2146
## Time      -0.5117 0.0584 -0.0836
## attr(,"Probability")
## [1] 0.95
## attr(,"class")
## [1] "confint.bas"
```



### Prediction

We can use the usual `predict` function that we used for `lm` objects to obtain prediction from the `BAS` object `crime.ZS`. However, since we have different models to choose from under the Bayesian framework, we need to first specify which particular model we use to obtain the prediction. For example, if we would like to use the Bayesian model averaging results for coefficients to obtain predictions, we would specify the `estimator` argument in the `predict` function like the following


```r
crime.BMA = predict(crime.ZS, estimator = "BMA", se.fit = TRUE)
```

The fitted values can be obtained using the `fit` attribute of `crime.BMA`. We have transposed the fitted values into a vector to better present all the values.


```r
fitted = crime.BMA$fit
as.vector(fitted)
```

```
##  [1] 6.662102 7.299052 6.178545 7.611169 7.054338 6.513008 6.784036 7.266359
##  [9] 6.629358 6.601242 7.054369 6.570461 6.472877 6.582381 6.557376 6.904836
## [17] 6.230285 6.809761 6.943485 6.962374 6.609043 6.429177 6.898151 6.777073
## [25] 6.405548 7.401339 6.019501 7.156779 7.089381 6.500144 6.208405 6.606137
## [33] 6.797999 6.820327 6.625154 7.028899 6.794072 6.362660 6.602801 7.045001
## [41] 6.548010 6.047404 6.930027 7.006628 6.236292 6.607844 6.830789
```

We may use these fitted values for further error calculations. We will talk about decision making on models and how to obtain predictions under different models in the next section.
