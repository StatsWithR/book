---
output:
  pdf_document: default
  html_document: default
---
## Comparing Independent  Means: Hypothesis Testing {#sec:indep-means}

In the previous section, we described Bayes factors for testing whether the mean difference of **paired** samples was zero. In this section, we will consider a slightly different problem -- we have two **independent** samples, and we would like to test the hypothesis that the means are different or equal.

\BeginKnitrBlock{example}
<span class="example" id="exm:birth-records"><strong>(\#exm:birth-records) </strong></span>We illustrate the testing of independent groups with data from a 2004 survey of birth records from North Carolina, which are available in the \texttt{statsr} package.

The variable of interest is \texttt{gained} -- the weight gain of mothers during pregnancy. We have two groups defined by the categorical variable, \texttt{mature}, with levels, younger mom and older mom.

**Question of interest**: Do the data provide convincing evidence of a difference between the average weight gain of older moms and the average weight gain of younger moms?
\EndKnitrBlock{example}

We will view the data as a random sample from two populations, older and younger moms. The two groups are modeled as:

\begin{equation}
\begin{aligned}
Y_{O,i} & \mathrel{\mathop{\sim}\limits^{\rm iid}} \textsf{N}(\mu + \alpha/2, \sigma^2) \\
Y_{Y,i} & \mathrel{\mathop{\sim}\limits^{\rm iid}} \textsf{N}(\mu - \alpha/2, \sigma^2)
\end{aligned}
(\#eq:half-alpha)
\end{equation}

The model for weight gain for older moms using the subscript $O$, and it assumes that the observations are independent and identically distributed, with a mean $\mu+\alpha/2$ and variance $\sigma^2$.

For the younger women, the observations with the subscript $Y$ are independent and identically distributed with a mean $\mu-\alpha/2$ and variance $\sigma^2$.

Using this representation of the means in the two groups, the difference in means simplifies to $\alpha$ -- the parameter of interest.

$$(\mu + \alpha/2)  - (\mu - \alpha/2) =  \alpha$$

You may ask, "Why don't we set the average weight gain of older women to $\mu+\alpha$, and the average weight gain of younger women to $\mu$?" We need the parameter $\alpha$ to be present in both $Y_{O,i}$ (the group of older women) and $Y_{Y,i}$ (the group of younger women).

We have the following competing hypotheses:

* $H_1: \alpha = 0 \Leftrightarrow$ The means are not different.
* $H_2: \alpha \neq 0 \Leftrightarrow$ The means are different.

In this representation, $\mu$ represents the overall weight gain for all women. (Does the model in Equation \@ref(eq:half-alpha) make more sense now?) To test the hypothesis, we need to specify prior distributions for $\alpha$ under $H_2$ (c.f. $\alpha = 0$ under $H_1$) and priors for $\mu,\sigma^2$ under both hypotheses.

Recall that the Bayes factor is the ratio of the distribution of the data under the two hypotheses.

$$\begin{aligned}
 \BF[H_1 : H_2] &=  \frac{p(\data \mid H_1)} {p(\data \mid H_2)} \\
  &= \frac{\iint p(\data \mid \alpha = 0,\mu,  \sigma^2 )p(\mu, \sigma^2 \mid H_1) \, d\mu \,d\sigma^2}
 {\int \iint p(\data \mid \alpha, \mu, \sigma^2) p(\alpha \mid \sigma^2) p(\mu, \sigma^2 \mid H_2) \, d \mu \, d\sigma^2 \, d \alpha}
\end{aligned}$$

As before, we need to average over uncertainty and the parameters to obtain the unconditional distribution of the data. Now, as in the test about a single mean, we cannot use improper or non-informative priors for $\alpha$ for testing.

Under $H_2$, we use the Cauchy prior for $\alpha$, or equivalently, the Cauchy prior on the standardized effect $\delta$ with the scale of $r$:

$$\delta = \alpha/\sigma^2 \sim \Ca(0, r^2)$$

Now, under both $H_1$ and $H_2$, we use the Jeffrey's reference prior on $\mu$ and $\sigma^2$:

$$p(\mu, \sigma^2) \propto 1/\sigma^2$$

While this is an improper prior on $\mu$, this does not suffer from the Bartlett's-Lindley's-Jeffreys' paradox as $\mu$ is a common parameter in the model in $H_1$ and $H_2$. This is another example of the Jeffreys-Zellner-Siow prior.

As in the single mean case, we will need numerical algorithms to obtain the Bayes factor. Now the following output illustrates testing of Bayes factors, using the Bayes inference function from the \texttt{statsr} package.


```r
library(statsr)
data(nc)
bayes_inference(y=gained, x=mature, data=nc,type='ht', 
                statistic='mean',  alternative='twosided', null=0,
                prior='JZS', r=1, method='theo', show_summ=FALSE)
```

```
## Hypotheses:
## H1: mu_mature mom  = mu_younger mom
## H2: mu_mature mom != mu_younger mom
## 
## Priors: P(H1) = 0.5  P(H2) = 0.5 
## 
## Results:
## BF[H1:H2] = 5.7162
## P(H1|data) = 0.8511 
## P(H2|data) = 0.1489 
## 
## Posterior summaries for under H2:
## 95% Cred. Int.: (-4.4281 , 0.8908)
```



\begin{center}\includegraphics{05-BFnormal-03-independent-means_files/figure-latex/bf-1} \end{center}

We see that the Bayes factor for $H_1$ to $H_2$ is about 5.7, with positive support for $H_1$ that there is no difference in average weight gain between younger and older women. Using equal prior probabilities, the probability that there is a difference in average weight gain between the two groups is about 0.15 given the data. Based on the interpretation of Bayes factors from Table \@ref(tab:jeffreys1961), this is in the range of "positive" (between 3 and 20).

To recap, we have illustrated testing hypotheses about population means with two independent samples, using a Cauchy prior on the difference in the means. One assumption that we have made is that **the variances are equal in both groups**. The case where the variances are unequal is referred to as the Behren-Fisher problem, and this is beyond the scope for this course. In the next section, we will look at another example to put everything together with testing and discuss summarizing results.
