## Decision Making Under Model Uncertainty

We are closing this chapter by presenting the last topic, decision making under model uncertainty. We have seen that under the Bayesian framework, we can use different prior distributions for coefficients, different model priors for models, and we can even use stochastic exploration methods for complex model selections. After selecting these coefficient priors and model priors, we can obtain the marginal posterior inclusion probability for each variable in the full model, which may provide some information about whether or not to include a particular variable in the model for further model analysis and predictions. With all the information presented in the results, which model would be the most appropriate model? 

In this section, we will talk about different methods for selecting models and decision making for posterior distributions and predictions. We will illustrate this process using the US crime data `UScrime` as an example and process it using the `BAS` package. 

We first prepare the data as in the last section and run `bas.lm` on the full model


```r
library(MASS)
data(UScrime)

# take the natural log transform on the variables except the 2nd column `So`
UScrime[, -2] = log(UScrime[, -2])

# run Bayesian linear regression
library(BAS)
crime.ZS =  bas.lm(y ~ ., data = UScrime,
                   prior = "ZS-null", modelprior = uniform()) 
```


### Model Choice

For Bayesian model choice, we start with the full model, which includes all the predictors. The uncertainty of selecting variables, or model uncertainty that we have been discussing, arises when we believe that some of the explanatory variables may be unrelated to the response variable. This corresponds to setting a regression coefficient $\beta_j$ to be exactly zero. We specify prior distributions that reflect our uncertainty about the importance of variables. We then update the model based on the data we obtained, resulting in posterior distributions over all models and the coefficients and variances within each model.

Now the question has become, how to select a single model from the posterior distribution and use it for furture inference? What are the objectives from inference?

**BMA Model**

We do have a single model, the one that is obtained by averaging all models using their posterior probabilities, the Bayesian model averaging model, or BMA. This is referred to as a hierarchical model and it is composed of many simpler models as building blocks. This represents the full posterior uncertainty after seeing the data. 

We can obtain the posterior predictive mean by using the weighted average of all of the predictions from each sub model

$$\hat{\mu} = E[\hat{Y}~|~\text{data}] = \sum_{M_m \in \text{ model space}}\hat{Y}\times p(M_m~|~\text{data}).$$
This prediction is the best under the squared error loss $L_2$. From `BAS`, we can obtain predictions and fitted values using the usual `predict` and `fitted` functions. To specify which model we use for these results, we need to include argument `estimator`.


```r
crime.BMA = predict(crime.ZS, estimator = "BMA")
mu_hat = fitted(crime.ZS, estimator = "BMA")
```

`crime.BMA`, the object obtained by the `predict` function, has additional slots storing results from the BMA model.


```r
names(crime.BMA)
```

```
##  [1] "fit"         "Ybma"        "Ypred"       "postprobs"   "se.fit"     
##  [6] "se.pred"     "se.bma.fit"  "se.bma.pred" "df"          "best"       
## [11] "bestmodel"   "prediction"  "estimator"
```

Plotting the two sets of fitted values, one obtained from the `fitted` function, another obtained from the `fit` attribute of the `predict` object `crime.BMA`, we see that they are in perfect agreement.


```r
# Load library and prepare data frame
library(ggplot2)
output = data.frame(mu_hat = mu_hat, fitted = crime.BMA$fit)

# Plot result from `fitted` function and result from `fit` attribute
ggplot(data = output, aes(x = mu_hat, y = fitted)) + 
  geom_point(pch = 16, color = "blue", size = 3) + 
  geom_abline(intercept = 0, slope = 1) + 
  xlab(expression(hat(mu[i]))) + ylab(expression(hat(Y[i])))
```



\begin{center}\includegraphics[width=0.7\linewidth]{08-MCMC-04-decisions_files/figure-latex/BMA-fit-1} \end{center}


**Highest Probability Model**

If our objective is to learn what is the most likely model to have generated the data using a 0-1 loss $L_0$, then the highest probability model (HPM) is optimal. 


```r
crime.HPM = predict(crime.ZS, estimator = "HPM")
```


The variables selected from this model can be obtained using the `bestmodel` attribute from the `crime.HPM` object. We can print out their names combining `bestmodel` in `crime.HPM` and `namesx` in `crime.ZS`


```r
crime.ZS$namesx[crime.HPM$bestmodel +1]
```

```
## [1] "Intercept" "M"         "Ed"        "Po1"       "NW"        "U2"       
## [7] "Ineq"      "Prob"      "Time"
```

We see that, except the intercept, which is always in any models, the highest probability model also includes `M`, percentage of males aged 14-24; `Ed`, mean years of schooling; `Po1`, police expenditures in 1960; `NW`, number of non-whites per 1000 people; `U2`, unemployment rate of urban males aged 35-39; `Ineq`, income inequlity; `Prob`, probability of imprisonment, and `Time`, average time in state prison.

To obtain the coefficients and their posterior means and posterior standard deviations, we can extract the model by using the `best` attribute of `crime.HPM` object. 


```r
# Obtain coefficients of all models
coef.crime.ZS = coef(crime.ZS)

# Select coefficients of HPM

# Posterior means of coefficients
coef.crime.ZS$conditionalmeans[crime.HPM$best, ]
```

```
##    Intercept            M           So           Ed          Po1 
##  6.724936198  1.489425882  0.000000000  0.000000000  0.000000000 
##          Po2           LF          M.F          Pop           NW 
##  0.000000000  1.244842537  0.116602482  0.002556687  0.174144505 
##           U1           U2          GDP         Ineq         Prob 
##  0.000000000  0.325391659  1.470700932  0.000000000  0.000000000 
##         Time 
## -0.027557352
```

```r
# Posterior standard deviation of coefficients
coef.crime.ZS$conditionalsd[crime.HPM$best, ]
```

```
##  Intercept          M         So         Ed        Po1        Po2 
## 0.04219189 0.72472849 0.00000000 0.00000000 0.00000000 0.00000000 
##         LF        M.F        Pop         NW         U1         U2 
## 0.88639575 2.39161199 0.06710837 0.04697363 0.00000000 0.22359886 
##        GDP       Ineq       Prob       Time 
## 0.30651514 0.00000000 0.00000000 0.19027051
```

We can also obtain the posterior probability of this model using

```r
postprob.HPM = crime.ZS$postprobs[crime.HPM$best]
postprob.HPM
```

```
## [1] 0.01824728
```

we see that this highest probability model has posterior probability of only 0.018. There are many models that have comparable posterior probabilities. So even this model has the highest posterior probability, we are still pretty unsure about whether it is the best model.

**Median Probability Model**

Another model that is frequently reported, is the median probability model (MPM). This model includes all predictors whose marginal posterior inclusion probabilities are greater than 0.5. If the variables are all uncorrelated, this will be the same as the highest posterior probability model. For a sequence of nested models such as polynomial regression with increasing powers, the median probability model is the best single model for prediction.  

However, since in the US crime example, `Po1` and `Po2` are highly correlated, we see that the variables included in MPM are slightly different than the variables included in HPM.


```r
crime.MPM = predict(crime.ZS, estimator = "MPM")
crime.ZS$namesx[crime.MPM$bestmodel +1]
```

```
## [1] "Intercept" "M"         "Ed"        "Po1"       "NW"        "U2"       
## [7] "Ineq"      "Prob"
```

As we see, this model only includes 7 variables, `M`, `Ed`, `Po1`, `NW`, `U2`, `Ineq`, and `Prob`. It does not include `Time` variable as in HPM. 

When there are correlated predictors in non-nexted models, MPM in general does well. However, if the correlations among variables increase, MPM may miss important variables as the correlations tend to dilute the posterior inclusing probabilities of related variables.  

To obtain the coefficients in the median probability model, we need to redo `bas.lm` to specify in `bestmodel` argument that we would like to keep only the variables with posterior inclusion probabilities greater than 0.5, and we would only want to have 1 model by setting `n.models = 1`. In this way, we will force other low probability variables not to show up in the model, and we will re-calculate the posterior means and standard deviations for the variables that are included in MPM.


```r
# Re-run regression and specify `bestmodel` and `n.models`
crime.ZS.MPM = bas.lm(y ~ ., data = UScrime,
                      prior = "ZS-null", modelprior = uniform(),
                      bestmodel = crime.ZS$probne0 > 0.5, n.models = 1)

# Obtain coefficients of MPM
coef(crime.ZS.MPM)
```

```
## 
##  Marginal Posterior Summaries of Coefficients: 
## 
##  Using  BMA 
## 
##  Based on the top  1 models 
##            post mean  post SD   post p(B != 0)
## Intercept   6.72494    0.02713   1.00000      
## M           1.46180    0.43727   1.00000      
## So          0.00000    0.00000   0.00000      
## Ed          2.30642    0.43727   1.00000      
## Po1         0.87886    0.16204   1.00000      
## Po2         0.00000    0.00000   0.00000      
## LF          0.00000    0.00000   0.00000      
## M.F         0.00000    0.00000   0.00000      
## Pop         0.00000    0.00000   0.00000      
## NW          0.08162    0.03743   1.00000      
## U1          0.00000    0.00000   0.00000      
## U2          0.31053    0.12816   1.00000      
## GDP         0.00000    0.00000   0.00000      
## Ineq        1.18815    0.28710   1.00000      
## Prob       -0.18401    0.06466   1.00000      
## Time        0.00000    0.00000   0.00000
```


**Best Predictive Model**

If our objective is prediction from a single model, the best choice is to find the model whose predictions are closet to those given by BMA. "Closest" could be based on squared error loss for predictions, or be based on any other loss functions. Unfortunately, there is no nice expression for this model. However, we can still calculate the loss for each of our sampled models to try to identify this best predictive model, or BPM.

Using the squared error loss, we find that the best predictive model is the one whose predictions are closest to BMA. 


```r
crime.BPM = predict(crime.ZS, estimator = "BPM")
crime.ZS$namesx[crime.BPM$bestmodel + 1]
```

```
##  [1] "Intercept" "M"         "So"        "Ed"        "Po1"      
##  [6] "Po2"       "M.F"       "NW"        "U2"        "Ineq"     
## [11] "Prob"
```

The best predictive model includes not only the 7 variables that MPM includes, but also `M.F`, number of males per 1000 females, and `Po2`, the police expenditures in 1959. 

Using the `se.fit = TRUE` option with `predict` we can calculate standard deviations for the predictions or for the mean. Then we can use this as input for the `confint` function for the prediction object. Here we only show the results of the first 20 data points.




```r
crime.BPM = predict(crime.ZS, estimator = "BPM", se.fit = TRUE)
crime.BPM.conf.fit = confint(crime.BPM, parm = "mean")
crime.BPM.conf.pred = confint(crime.BPM, parm = "pred")
cbind(crime.BPM$fit, crime.BPM.conf.fit, crime.BPM.conf.pred)
##                    2.5%    97.5%     mean     2.5%    97.5%     pred
##  [1,] 6.668988 6.513238 6.824738 6.668988 6.258715 7.079261 6.668988
##  [2,] 7.290854 7.151787 7.429921 7.290854 6.886619 7.695089 7.290854
##  [3,] 6.202166 6.039978 6.364354 6.202166 5.789406 6.614926 6.202166
##  [4,] 7.661307 7.490608 7.832006 7.661307 7.245129 8.077484 7.661307
##  [5,] 7.015570 6.847647 7.183493 7.015570 6.600523 7.430617 7.015570
##  [6,] 6.469547 6.279276 6.659818 6.469547 6.044966 6.894128 6.469547
##  [7,] 6.776133 6.555130 6.997135 6.776133 6.336920 7.215346 6.776133
##  [8,] 7.299560 7.117166 7.481955 7.299560 6.878450 7.720670 7.299560
##  [9,] 6.614927 6.482384 6.747470 6.614927 6.212890 7.016964 6.614927
## [10,] 6.596912 6.468988 6.724836 6.596912 6.196374 6.997449 6.596912
## [11,] 7.032834 6.877582 7.188087 7.032834 6.622750 7.442918 7.032834
## [12,] 6.581822 6.462326 6.701317 6.581822 6.183896 6.979748 6.581822
## [13,] 6.467921 6.281998 6.653843 6.467921 6.045271 6.890571 6.467921
## [14,] 6.566239 6.403813 6.728664 6.566239 6.153385 6.979092 6.566239
## [15,] 6.550129 6.388987 6.711270 6.550129 6.137779 6.962479 6.550129
## [16,] 6.888592 6.746097 7.031087 6.888592 6.483166 7.294019 6.888592
## [17,] 6.252735 6.063944 6.441526 6.252735 5.828815 6.676654 6.252735
## [18,] 6.795764 6.564634 7.026895 6.795764 6.351369 7.240160 6.795764
## [19,] 6.945687 6.766289 7.125086 6.945687 6.525866 7.365508 6.945687
## [20,] 7.000331 6.840374 7.160289 7.000331 6.588442 7.412220 7.000331
## [...]

```

We can use similar method as in HPM to find the coefficients of BPM

```r
# Posterior mean
coef.crime.ZS$conditionalmeans[crime.BPM$best,]
```

```
##  Intercept          M         So         Ed        Po1        Po2 
##  6.7249362  1.4642381  0.0000000  0.0000000  1.0179945  0.0000000 
##         LF        M.F        Pop         NW         U1         U2 
##  0.0000000  0.0000000  0.0000000  0.0000000  0.1050230  0.0000000 
##        GDP       Ineq       Prob       Time 
##  0.9092126  1.5817768 -0.2319145 -0.2789096
```

```r
# Posterior standard deviation
coef.crime.ZS$conditionalsd[crime.BPM$best,]
```

```
##  Intercept          M         So         Ed        Po1        Po2 
## 0.03229012 0.50690367 0.00000000 0.00000000 0.16937453 0.00000000 
##         LF        M.F        Pop         NW         U1         U2 
## 0.00000000 0.00000000 0.00000000 0.00000000 0.18641024 0.00000000 
##        GDP       Ineq       Prob       Time 
## 0.42192265 0.32635089 0.10123170 0.16467504
```


After discussing all 4 different models, let us compare their prediction results. 


```r
# Set plot settings
par(cex = 1.8, cex.axis = 1.8, cex.lab = 2, mfrow = c(2,2), mar = c(5, 5, 3, 3),
    col.lab = "darkgrey", col.axis = "darkgrey", col = "darkgrey")

# Load library and plot paired-correlations
library(GGally)
ggpairs(data.frame(HPM = as.vector(crime.HPM$fit),  
                   MPM = as.vector(crime.MPM$fit),  
                   BPM = as.vector(crime.BPM$fit),  
                   BMA = as.vector(crime.BMA$fit))) 
```



\begin{center}\includegraphics[width=0.7\linewidth]{08-MCMC-04-decisions_files/figure-latex/paired-cor-1} \end{center}

From the above paired correlation plots, we see that the correlations among them are extremely high. As expected, the single best predictive model (BPM) has the highest correlation with MPM, with a correlation of 0.998. However, the highest posterior model (HPM) and the Bayesian model averaging model (BMA) are nearly equally as good.

 
### Prediction with New Data

Using the `newdata` option in the `predict` function, we can obtain prediction from a new data set. Here we pretend that `UScrime` is an another new data set, and we use BMA to obtain the prediction of new observations. Here we only show the results of the first 20 data points.


```r
BMA.new = predict(crime.ZS, newdata = UScrime, estimator = "BMA",
                  se.fit = TRUE, nsim = 10000)
crime.conf.fit.new = confint(BMA.new, parm = "mean")
crime.conf.pred.new = confint(BMA.new, parm = "pred")

# Show the combined results compared to the fitted values in BPM
cbind(crime.BPM$fit, crime.conf.fit.new, crime.conf.pred.new)
##                    2.5%    97.5%     mean     2.5%    97.5%     pred
##  [1,] 6.668988 6.511908 6.814620 6.661770 6.256785 7.070667 6.661770
##  [2,] 7.290854 7.128951 7.455601 7.298827 6.882325 7.717802 7.298827
##  [3,] 6.202166 5.953686 6.407157 6.179308 5.715855 6.598747 6.179308
##  [4,] 7.661307 7.380899 7.832700 7.610585 7.145835 8.062224 7.610585
##  [5,] 7.015570 6.853424 7.259395 7.054238 6.623308 7.483381 7.054238
##  [6,] 6.469547 6.282326 6.733552 6.514064 6.056658 6.962595 6.514064
##  [7,] 6.776133 6.501995 7.065373 6.784846 6.305445 7.270351 6.784846
##  [8,] 7.299560 7.045325 7.481729 7.266344 6.831083 7.706456 7.266344
##  [9,] 6.614927 6.484771 6.783354 6.629448 6.213834 7.055457 6.629448
## [10,] 6.596912 6.459036 6.734196 6.601246 6.206266 7.017514 6.601246
## [11,] 7.032834 6.868390 7.241408 7.055003 6.633615 7.500111 7.055003
## [12,] 6.581822 6.427783 6.718648 6.570625 6.175967 7.004224 6.570625
## [13,] 6.467921 6.216010 6.723663 6.472327 6.014310 6.936410 6.472327
## [14,] 6.566239 6.393780 6.761470 6.582374 6.166734 7.023871 6.582374
## [15,] 6.550129 6.359419 6.761501 6.556880 6.139225 7.000841 6.556880
## [16,] 6.888592 6.744461 7.064665 6.905017 6.480618 7.313643 6.905017
## [17,] 6.252735 5.987745 6.468435 6.229073 5.783551 6.694666 6.229073
## [18,] 6.795764 6.543715 7.101233 6.809572 6.340140 7.281983 6.809572
## [19,] 6.945687 6.756101 7.125568 6.943294 6.512679 7.391876 6.943294
## [20,] 7.000331 6.783661 7.147549 6.961980 6.525086 7.389334 6.961980
## [...]

```

## Summary

In this chapter, we have introduced one of the common stochastic exploration methods, Markov Chain Monte Carlo, to explore the model space to obtain approximation of posterior probability of each model when the model space is too large for theoretical enumeration. We see that model selection is very sensitive to the prior distributions of coefficients. Therefore, besides the reference prior, we have also introduced the Zellner's $g$-prior. To solve the paradoc problems, we have improved this Zellner's $g$-prior by imposing relationship between the scalar $g$ and the sample size $n$, which leads to other priors, such as the unit information $g$-prior, the Zellner-Siow cauchy prior, and the hyper-$g/n$ prior. 

We later have demonstrated a multiple linear regression process using `BAS` package and the US crime data `UScrime`. We have diagnosed the results using the Zellner-Siow cauchy prior, and have tried to understand the importance of variables. 
Finally, we have compared the prediction results from different models, such as the ones from Bayesian model average (BMA), the highest probability model (HPM), the median probability model (MPM), and the best predictive model (BPM). For the comparison, we have used the Zellner-Siow cauchy prior. But of course there is not one single best prior that is the best overall. If you do have prior information about a variable, you should include it. If you expect that there should be many predictors related to the response variable $Y$, but that each has a small effect, an alternate prior may be better. Also, think critically about whether model selection is important. If you believe that all the variables should be relevant but are worried about over fitting, there are alternative priors that will avoid putting probabilities on coefficients that are exactly zero and will still prevent over fitting by shrinkage of coefficients to prior means. Examples include the Bayesian lasso or Bayesian horseshoe.

There are other forms of model uncertainty that you may want to consider, such as linearity in the relationship between the predictors and the response, uncertainty about the presence of outliers, and uncertainty about the distribution of the response. These forms of uncertainty can be incorporated by expanding the models and priors similar to what we have covered here. 

Multiple linear regression is one of the most widely used statistical methods, however, this is just the tip of the iceberg of what you can do with Bayesian methods. 
