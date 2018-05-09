---
output:
  html_document: default
  pdf_document: default
---
## Inference after Testing

In this section, we will work through another example for comparing two means using both hypothesis tests and interval estimates, with an informative prior.
We will also illustrate how to adjust the credible interval after testing.

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:smoking"><strong>(\#exm:smoking) </strong></span>We will use the North Carolina survey data to examine the relationship
between infant birth weight and whether the mother smoked during pregnancy. The response variable, \texttt{weight}, is the birth weight of the baby in pounds. The categorical variable \texttt{habit} provides the status of the mother as a smoker or non-smoker.</div>\EndKnitrBlock{example}

We would like to answer two questions:
  
1. Is there a difference in average birth weight between the two groups?

2. If there is a difference, how large is the effect?

As before, we need to specify models for the data and priors. We treat the data as a random sample for the two populations, smokers and non-smokers.

The birth weights of babies born to non-smokers, designated by a subgroup $N$, are assumed to be independent and identically distributed from a normal distribution with mean $\mu + \alpha/2$, as in Section \@ref(sec:indep-means).

$$Y_{N,i} \iid \No(\mu + \alpha/2, \sigma^2)$$

While the birth weights of the babies born to smokers, designated by the subgroup $S$, are also assumed to have a normal distribution, but with mean $\mu - \alpha/2$.

$$Y_{S,i} \iid \No(\mu - \alpha/2, \sigma^2)$$

The difference in the average birth weights is the parameter $\alpha$, because

$$(\mu + \alpha/2) - (\mu - \alpha/2) =  \alpha$$.

The hypotheses that we will test are $H_1:  \alpha = 0$  versus $H_2:  \alpha \ne 0$.

We will still use the Jeffreys-Zellner-Siow Cauchy prior. However, since we may expect the standardized effect size to not be as strong, we will use a scale of $r = 0.5$ rather than 1.

Therefore, under $H_2$, we have  
$$\delta = \alpha/\sigma \sim \Ca(0, r^2), \text{ with } r = 0.5.$$

Under both $H_1$ and $H_2$, we will use the reference priors on $\mu$ and $\sigma^2$:

$$\begin{aligned}
p(\mu) &\propto 1 \\
p(\sigma^2) &\propto 1/\sigma^2
\end{aligned}$$

The input to the base inference function is similar, but now we will specify that $r = 0.5$.


```r
library(statsr)
data(nc)
out =bayes_inference(y=weight, x=habit, data=nc,type='ht', null=0,
                     statistic='mean',  alternative='twosided',
                     prior='JZS', r=.5, method='sim', show_summ=FALSE)
```

```
## Hypotheses:
## H1: mu_nonsmoker  = mu_smoker
## H2: mu_nonsmoker != mu_smoker
## 
## Priors: P(H1) = 0.5  P(H2) = 0.5 
## 
## Results:
## BF[H2:H1] = 1.4402
## P(H1|data) = 0.4098 
## P(H2|data) = 0.5902 
## 
## Posterior summaries for under H2:
## 95% Cred. Int.: (0.017 , 0.5741)
```



\begin{center}\includegraphics{05-BFnormal-04-inference_files/figure-latex/BF-NC-1} \end{center}

We see that the Bayes factor is 1.44, which weakly favors there being a difference in average birth weights for babies whose mothers are smokers versus mothers who did not smoke. Converting this to a probability, we find that there is about a 60% chance of the average birth weights are different.

While looking at evidence of there being a difference is useful, Bayes factors and posterior probabilities do **not** convey any information about the magnitude of the effect. Reporting a credible interval or the complete posterior distribution is more relevant for quantifying the magnitude of the effect.

Using the \texttt{bayes$\_$inference} function, we can generate samples from the posterior distribution under $H_2$ using the \texttt{type='ci'} option.


```r
out.ci = bayes_inference(y=weight, x=habit, data=nc, type='ci',
                         statistic='mean', prior='JZS', mu_0=0,
                         r=.5, method='sim', verbose=FALSE)
print(out.ci$summary, digits=2)
```

```
##                             2.5%     25%    50%     75%   97.5%
## overall mean               6.855    6.95    7.0    7.04 7.1e+00
## mu_nonsmoker - mu_smoker   0.028    0.20    0.3    0.39 5.7e-01
## sigma^2                    2.070    2.19    2.3    2.33 2.5e+00
## effect size                0.019    0.14    0.2    0.26 3.8e-01
## n_0                      162.885 1905.03 4653.5 9398.61 2.5e+04
```

The 2.5 and 97.5 percentiles for the difference in the means provide a 95% credible interval of 0.023 to 0.57 pounds for the difference in average birth weight. The MCMC output shows not only summaries about the difference in the mean $\alpha$, but the other parameters in the model.

In particular, the Cauchy prior arises by placing a gamma prior on $n_0$ and the conjugate normal prior. This provides quantiles about $n_0$ after updating with the current data.

The row labeled effect size is the standardized effect size $\delta$, indicating that the effects are indeed small relative to the noise in the data.


```r
library(ggplot2)
out = bayes_inference(y=weight, x=habit, data=nc,type='ht',
                statistic='mean',  alternative='twosided',
                prior='JZS', null=0, r=.5, method='theo',
                show_summ=FALSE, show_res=FALSE, show_plot=TRUE)
```

\begin{figure}

{\centering \includegraphics{05-BFnormal-04-inference_files/figure-latex/BF-NC-plot-1} 

}

\caption{Estimates of effect under H2}(\#fig:BF-NC-plot)
\end{figure}

Figure \@ref(fig:BF-NC-plot) shows the posterior density for
the difference in means, with the 95% credible interval indicated by the shaded area. Under $H_2$, there is a 95% chance that the average birth weight of babies born to non-smokers is 0.023 to 0.57 pounds higher than that of babies born to smokers.

The previous statement assumes that $H_2$ is true and is a conditional probability statement. In mathematical terms, the statement is equivalent to

$$P(0.023 < \alpha < 0.57 \mid \data, H_2) =  0.95$$

However, we still have quite a bit of uncertainty based on the current data, because given the data, the probability of $H_2$ being true is 0.59.

$$P(H_2 \mid \data) = 0.59$$

Using the law of total probability, we can compute the probability that $\mu$ is between 0.023 and 0.57 as below:

$$\begin{aligned}
& P(0.023 < \alpha < 0.57 \mid \data) \\
= & P(0.023 < \alpha < 0.57 \mid \data, H_1)P(H_1 \mid \data)  + P(0.023 < \alpha < 0.57 \mid \data, H_2)P(H_2 \mid \data) \\
= & I( 0 \text{ in CI }) P(H_1 \mid \data)  + 0.95 \times P(H_2 \mid \data) \\
= & 0 \times 0.41 + 0.95 \times 0.59 = 0.5605
\end{aligned}$$

Finally, we get that the probability that $\alpha$ is in the interval, given the data, averaging over both hypotheses, is roughly 0.56. The unconditional statement is the average birth weight of babies born to nonsmokers is 0.023 to 0.57 pounds higher than that of babies born to smokers with probability 0.56. This adjustment addresses the posterior uncertainty and how likely $H_2$ is.

To recap, we have illustrated testing, followed by reporting credible intervals, and using a Cauchy prior distribution that assumed smaller standardized effects. After testing, it is common to report credible intervals conditional on $H_2$. We also have shown how to adjust the probability of the interval to reflect our posterior uncertainty about $H_2$. In the next chapter, we will turn to regression models to incorporate continuous explanatory variables.
