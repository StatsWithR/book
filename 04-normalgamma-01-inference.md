## The Normal-Gamma Conjugate Family {#sec:normal-gamma}


You may take the safety of your drinking water for granted, however, residents of Flint,  Michigan were outraged over reports that the levels of a contaminant known as __TTHM__ exceeded federal allowance levels in 2014.   TTHM stands for  total trihalomethanes, a group of chemical compounds first identified in drinking water in the 1970’s. Trihalomethanes are formed as a by-product  from the reaction of chlorine or bromine with organic matter present in the water being disinfected for drinking. THMs have been associated through epidemiological studies with some adverse health effects and many are considered carcinogenic. In the United States, the EPA limits the total concentration of the four chief constituents (chloroform, bromoform, bromodichloromethane, and dibromochloromethane), referred to as total trihalomethanes (TTHM), to 80 parts per billion in treated water.

Since violations are based on annual running averages, we are interested in inference about the mean TTHM level based on measurements taken from samples.  

In Section \@ref(sec:normal-normal) we described the normal-normal conjugate  family for inference about an unknown mean $\mu$ when the data $Y_1, Y_2, \ldots, Y_n$ were assumed to be a random sample of size $n$ from a normal population  with a known standard deviation $\sigma$, however, it is more common in practice to have data where the variability of observations is unknown, as in the example with TTHM. Conceptually, Bayesian inference for two (or more) parameters is not any different from the case with one parameter.  As both $\mu$ and $\sigma^2$ unknown, we will need to specify a **joint** prior distribution, $p(\mu, \sigma^2)$ to describe our prior uncertainty about them. As before, Bayes Theorem leads to the posterior distribution for $\mu$ and $\sigma^2$ given the observed data to take the form

\begin{equation}
p(\mu, \sigma^2 \mid y_1, \ldots, y_n)  = 
\frac{p(y_1, \ldots, y_n \mid \mu, \sigma^2) \times
p(\mu, \sigma^2)}
{\text{normalizing constant}}. 
\end{equation}
The **likelihood function** for $\mu, \sigma^2$ is proportional to the sampling distribution of the data,  ${\cal L}(\mu, \sigma^2) \propto p(y_1, \ldots, y_n \mid \mu, \sigma^2)$ so that the posterior distribution can be re-expressed in proportional form
\begin{equation}
p(\mu, \sigma^2 \mid y_1, \ldots, y_n)  \propto {\cal L}(\mu, \sigma^2) p(\mu, \sigma^2).
\end{equation}


As in the earlier chapters, conjugate priors are appealing as there are nice expressions for updating the prior to obtain the posterior distribution using summaries of the data. In the case of two parameters or more parameters a conjugate pair is a sampling model for the data  and a joint prior distribution for the unknown parameters such that the joint posterior distribution is in the same family of distributions as the prior distribution.  In this case our sampling model is built on the assumption that the data are a random sample of size $n$ from a normal population with mean $\mu$ and variance $\sigma^2$, expressed in  shorthand as

$$\begin{aligned}
Y_1, \ldots Y_n  \iid
\textsf{Normal}(\mu, \sigma^2) 
\end{aligned}$$
where the  'iid' above the distribution symbol '$\sim$'  indicates that each of the observations are **i**ndependent of the others (given $\mu$ and $\sigma^2$) and are **i**dentically **d**istributed.  Under this assumption, the sampling distribution of the data is the product of independent normal distributions with mean $\mu$ and variance $\sigma^2$,
\begin{equation}
p(y_1, \ldots, y_n \mid \mu, \sigma^2) = \prod_{i = 1}^n
\frac{1}{\sqrt{2 \pi \sigma^2}}
e^{\left\{- \frac{1}{2} \left(\frac{y_i - \mu}{\sigma}\right)^2\right\}}
\end{equation}
which, after some algebraic manipulation and simplification, leads to a
likelihood function for $\mu$ and $\sigma^2$ that is proportional to
$$
\begin{aligned}
{\cal L}(\mu, \sigma^2) \propto 
(\sigma^2)^{-n/2}  \times \exp{ \left\{   
-\frac{1}{2} \frac{\sum_{i = 1}^n(y_i - \bar{y})^2 }{\sigma^2}
\right\}}  & \times 
\exp{ \left\{   
-\frac{1}{2} \frac{n (\bar{y} - \mu)^2 }{\sigma^2} \right\}} \\
  \text{function of $\sigma^2$ and data} & \times
  \text{function of $\mu$, $\sigma^2$ and data}
\end{aligned}
$$ 
which depends on the data only through the sum of squares $\sum_{i = 1}^n(y_i - \bar{y})^2$ (or equivalently the sample variance $s^2 = \sum_{i = 1}^n(y_i - \bar{y})^2/(n-1)$) and the sample mean $\bar{y}$. 
From the expression for the likelihood, we can see that the likelihood factors into two pieces: a term that is a function of $\sigma^2$ and the data; and a term that involves $\mu$, $\sigma^2$ and the data.


Based on the factorization in the likelihood and the fact that any joint distribution for $\mu$ and $\sigma^2$ can be expressed as
$$
p(\mu, \sigma^2) = p(\mu \mid \sigma^2) \times p(\sigma^2)
$$
as the product of a **conditional distribution** for $\mu$ given $\sigma^2$ and a **marginal distribution** for $\sigma^2$, this suggests that the posterior distribution should factor as the product of two conjugate distributions. Perhaps not surprisingly, this is indeed the case.

### Conjugate Prior for $\mu$ and $\sigma^2$

In Section \@ref(sec:normal-normal), we found that for normal data, the conjugate prior distribution for $\mu$ when the standard deviation $\sigma$ was known was a normal distribution.  We will build on this to specify a conditional prior distribution for $\mu$ as a normal distribution
\begin{equation}
\mu \mid \sigma^2   \sim  \textsf{N}(m_0, \sigma^2/n_0)
(\#eq:04-conjugate-normal)
\end{equation}
with hyper-parameters $m_0$, the prior mean for $\mu$, and $\sigma^2/n_0$ the prior variance.  While previously we represented the prior variance as a fixed constant, $\tau^2$, in this case we will replace $\tau^2$ with a multiple of $\sigma^2$.    Because $\sigma$ has the same units as the data, the presence of $\sigma$ in the prior variance automatically scales the prior for $\mu$ based on the same units.  This is important, for example, if we were to change the measurement units from inches to centimeters or seconds to hours, as the prior will be re-scaled automatically. The hyper-parameter $n_0$ is unitless, but is used to express our prior precision about $\mu$ relative to the level of "noise", captured by $\sigma^2$, in the data.  Larger values of $n_0$ indicate that we know the mean with more precision (relative to the variability in observations) with smaller values indicating less precision or more uncertainty.  We will see later how the hyper-parameter $n_0$ may be interpreted as a prior sample size.  Finally, while we could use a fixed value $\tau^2$ as the prior variance in a conditional conjugate prior for $\mu$ given $\sigma^2$,  that does not lead to a joint conjugate prior for $\mu$ and $\sigma^2$.

As $\sigma^2$ is unknown, a Bayesian would use a
prior distribution to describe the uncertainty about the variance before seeing data.  Since the variance is non-negative, continuous, and with no upper limit,  based on the distributions that we have seen so far a gamma distribution might appear to be a candidate  prior for the variance,. However, that choice does not lead to a posterior distribution in the same family or that is recognizable as any common distribution. It turns out that the the inverse of the variance, which is known as the precision, has a conjugate gamma prior distribution.

For simplification let's express the precision (inverse variance) as a new parameter,  $\phi = 1/\sigma^2$.  Then the conjugate prior for $\phi$,
\begin{equation}
\phi \sim \textsf{Gamma}\left(\frac{v_0}{2}, \frac{v_0 s^2_0}{2} \right)
(\#eq:04-conjugate-gamma)
\end{equation}
is a gamma distribution with  shape parameter $v_0/2$ and  **rate** parameter of ${v_0 s^2_0}/{2}$. Given the connections between the gamma distribution and the Chi-Squared distribution,  the hyper-parameter $v_0$ may be interpreted as the prior degrees of freedom.  The hyper-parameter $s^2_0$ may be interpreted as a prior variance or initial prior estimate for $\sigma^2$. Equivalently, we may say that the inverse of the variance has a 
$$1/\sigma^2 \sim \textsf{Gamma}(v_0/2, s^2_0 v_0/2)$$

gamma distribution to avoid using a new symbol \footnote{In some other references, you will see that $\sigma^2$ will have an inverse gamma distribution.  Rather than introduce an additional distribution for the inverse-gamma, we will restrict our attention to the gamma distribution since the inverse-gamma is equivalent to saying that the inverse of $\sigma^2$ has a gamma distribution and `R` has support for generating random variables from the gamma that we will need in later sections.}. Together the conditional normal distribution for $\mu$ given $\sigma^2$ in \@ref(eq:04-conjugate-normal)  and the marginal gamma distribution for $\phi$ in \@ref(eq:04-conjugate-gamma) lead to a joint distribution for the pair $(\mu, \phi)$ that we will call the normal-gamma family of distributions:
\begin{equation}(\mu, \phi) \sim \textsf{NormalGamma}(m_0, n_0, s^2_0, v_0)
(\#eq:04-conjugate-normal-gamma)
\end{equation}
with the four hyper-parameters $m_0$, $n_0$, $s^2_0$, and $v_0$.

We can obtain the density for the \textsf{Normal-Gamma}($m_0, n_0, \nu_0, s^2_0$) family of distributions for $\mu, \phi$ by multiplying the conditional normal distribution for $\mu$ times the marginal gamma distribution for $\phi$:
\begin{equation}
p(\mu, \phi) = \frac{(n_0 \phi)^{1/2}} {\sqrt{2\pi}} e^{- \frac{\phi n_0}{2} (\mu -m_0)^2} \frac{1}{\Gamma(\nu_0/2)} (\nu_0 s^2_0 )^{\nu_0/2 -1} e^{- \phi \frac{\nu_0 s^2_0} {2}}
\label{eq:NG}
\end{equation}


The joint conjugate prior has simple rules for updating the prior hyper-parameters given new data to obtain the posterior hyper-parameters due to conjugacy.

### Conjugate Posterior Distribution

As a conjugate family, the posterior
distribution of the pair of parameters ($\mu, \phi$) is in the same family as the prior distribution when the sample data arise from a normal distribution, that is the posterior is also normal-gamma 
\begin{equation}
(\mu, \phi) \mid \text{data} \sim \textsf{NormalGamma}(m_n, n_n, s^2_n, v_n)
\end{equation}
where the subscript $n$ on the
hyper-parameters indicates the updated values after seeing the $n$ observations from the sample data. One attraction of conjugate families is there are relatively simple updating rules for obtaining the new hyper-parameters:
\begin{eqnarray*}
m_n & = & \frac{n \bar{Y} + n_0 m_0} {n + n_0}  \\
& \\
n_n & = & n_0 + n  \\
v_n & = & v_0 + n  \\
s^2_n & =  & \frac{1}{v_n}\left[ s^2 (n-1) + s^2_0 v_0 + \frac{n_0 n}{n_n} (\bar{y} - m_0)^2 \right]. 
\end{eqnarray*}
Let's look more closely to try to understand the updating rules.
The updated hyper-parameter $m_n$ is the posterior mean for  $\mu$; it is also the mode and median.  The posterior mean $m_n$ is a weighted average of the sample mean $\bar{y}$ and prior mean $m_0$ with weights $n/(n + n_0$ and $n_0/(n + n_0)$ that are proportional to the precision in the data, $n$, and the prior  precision, $n_0$, respectively. 

The posterior sample size $n_n$ is the sum of the prior sample
size $n_0$ and the sample size $n$, representing the combined precision after seeing the data for the posterior distribution for $\mu$.  The posterior degrees of freedom $v_n$ are also increased by adding the  sample size $n$ to the prior degrees of freedom $v_0$.

Finally, the posterior variance hyper-parameter $s^2_n$ combines three sources of information about $\sigma^2$ in terms of sums of squared deviations. The first term in
the square brackets is the sample variance times the sample degrees of
freedom, $s^2 (n-1) = \sum_{i=1}^n (y_i - \bar{y})^2$, which is the sample sum of squares. Similarly, we may view the second term as a sum of squares based on prior data, where $s^2_0$ was an estimate of $\sigma^2$. The squared difference of the sample mean and prior mean in the last term also provides an estimate of $\sigma^2$, where a large value of $(\bar{y} - \mu_0)^2$ increases the posterior sum of squares $v_n s^2_n$.  
If the sample mean is far from our prior mean, this increases  the probability that $\sigma^2$ is large.  Adding these three sum of squares provides the posterior  sum of square, and dividing by the posterior 
posterior degrees of freedom we obtain  the new hyper-parameter $s^2_n$, which is an estimate of $\sigma^2$ combining the sources of variation from the prior and the data.

The joint posterior distribution for the pair $\mu$ and
$\phi$ 
$$(\mu, \phi) \mid \data \sim \NoGa(m_n, n_n, s^2_n, v_n)$$
is in the normal-gamma family, and  is equivalent to a **hierarchical model** specified in two stages: in the 
first stage of the hierarchy  the inverse variance or precision marginally has a gamma distribution,
$$
1/\sigma^2 \mid \data   \sim   \Ga(v_n/2, s^2_n v_n/2) 
$$
and in the second stage,  $\mu$ given $\sigma$

$$\mu \mid \data, \sigma^2  \sim  \No(m_n, \sigma^2/n_n)$$
has a conditional normal distribution.  We will see in the next chapter how this representation is convenient for generating samples from the posterior distribution.


### Marginal Distribution for $\mu$: Student $t$

The joint normal-gamma posterior summarizes our current knowledge about $\mu$ and $\sigma^2$, however, we are generally interested in inference about $\mu$ unconditionally 
as $\sigma^2$ is unknown. This marginal inference requires the unconditional or marginal distribution of $\mu$ that `averages' over the uncertainty in $\sigma$. For continuous variables like $\sigma$, this averaging is performed by integration leading to a Student $t$ distribution.  

The *standardized Student $t$-distribution*  $\St_\nu$ with $\nu$ degrees of freedom is defined to be
$$ p(t) = \frac{1}{\sqrt{\pi\nu}}\frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})}\left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}} $$ 
where the $\Gamma(\cdot)$ is the Gamma function defined earlier in Equation \@ref(eq:gamma-function). The standard Student's $t$-distribution is centered at 0 (the location parameter), with a scale parameter equal to 1, like in a standard normal, however, there is an additional parameter, $\nu$, the degrees of freedom parameter.

The Student $t$ distribution is similar to the normal distribution as it is symmetric about the center and bell shaped, however, the __tails__ of the distribution are fatter or heavier than the normal distribution   and therefore, it is a little "shorter" in the middle as illustrated in Figure \@ref(fig:density)


![(\#fig:t-density)Standard normal and Student t densities.](04-normalgamma-01-inference_files/figure-latex/t-density-1.pdf) 

Similar to the normal distribution, we can obtain other Student $t$ distributions by changing the center of the distribution and changing the scale.  A Student t distribution with a location $m$ and scale $s$ with $v$ degrees of freedom is denoted as  $\St(v, m, s^2)$, with the standard Student t as a special case, $\St(\nu, 0, 1)$.

The  density for a $X \sim \St(v, m, s^2)$ random variable is
\begin{equation}
p(x) =\frac{\Gamma\left(\frac{v + 1}{2} \right)}
{\sqrt{\pi v} s \,\Gamma\left(\frac{v}{2} \right)}
\left(1 + \frac{1}{v}\left(\frac{x - m} {s} \right)^2 \right)^{-\frac{v+1}{2}} 
(\#eq:Student-t-density)
\end{equation}
and by subtracting the location $m$ and dividing by the scale 
$s$:
$$ \frac{X - m}{s} \equiv t \sim \St(v, 0 , 1)  $$
we can obtain the distribution of the standardized Student $t$ distribution with degrees of freedom $v$, location  $0$ and scale $1$. This latter representation allows us to use standard statistical functions for posterior inference such as finding credible intervals.

We are now ready for our main result for the marginal distribution for $\mu$. 
::: {.definition #unnamed-chunk-1}
If $\mu$ and $1/\sigma^2$ have a $\textsf{NormalGamma}(m_n, n_n, v_n, s^2_n)$ posterior distribution, then 
$\mu$ given the data has a \index{Student t distribution} distribution, $\St(v_n, m_n, s^2_n/n_n)$, expressed as
$$ \mu \mid \data \sim \St(v_n, m_n, s^2_n/n_n)  $$ 
with degrees of freedom $v_n$, 
location parameter, $m_n$, and squared scale parameter, $s^2_n/n_n$, that is the
posterior variance parameter divided by the posterior sample size.  

:::

The parameters $m_n$ and $s^2_n$ play similar roles in determining the center and spread of the distribution, as in the normal distribution, however,  as Student $t$ distributions with degrees of freedom less than 3 do not have a mean or variance, the parameter $m_n$ is called the location or center of the distribution and the $s_n/\sqrt{n}$ is the scale.


Let's use this result to find credible intervals for $\mu$.





### Credible Intervals for $\mu$

To find a credible interval for the mean $\mu$, we will use the marginal posterior distribution for $\mu$ as illustrated in  Figure \@ref(fig:tapwater-post-mu).
Since the Student $t$ distribution of $\mu$ is unimodal and symmetric, the shortest 95 percent credible interval or the **Highest Posterior Density** interval, HPD for short,
is the  interval given by the dots at the
lower endpoint L and upper endpoint U where  the heights of the density at L and U are equal and all other values for $\mu$ have higher posterior density. The probability that $\mu$ is in the interval (L, U) (the shaded area) equals the desired probability, e.g. 0.95 for a 95% credible interval.
\begin{figure}

{\centering \includegraphics{04-normalgamma-01-inference_files/figure-latex/tapwater-post-mu-1} 

}

\caption{Highest Posterior Density region.}(\#fig:tapwater-post-mu)
\end{figure}



Using the standardized Student $t$ distribution and some algebra, these values are
$$
\begin{aligned}
  L & =  m_n + t_{0.025}\sqrt{s^2_n/n_n}    \\
  U & =  m_n + t_{0.975}\sqrt{s^2_n/n_n}
\end{aligned}
$$
or the posterior mean (our point estimate) plus quantiles of the standard $t$ distribution times the scale.  Because of the symmetry in the Student $t$ distribution, the credible interval for $\mu$ is $m_n \pm t_{0.975}\sqrt{s^2_n/n_n}$, which is similar to the expressions for confidence intervals for the mean.

### Example: TTHM in Tapwater  {#sec:tapwater}
A municipality in North Carolina is interested in estimating the levels of TTHM in their drinking water.   The data can be loaded from the `statsr` package in `R`, where the variable of interest, `tthm` is measured in parts per billion.



```r
library(statsr)
data(tapwater)
glimpse(tapwater)
```

```
## Rows: 28
## Columns: 6
## $ date       <fct> 2009-02-25, 2008-12-22, 2008-09-25, 2008-05-14, 2008-04-14,~
## $ tthm       <dbl> 34.38, 39.33, 108.63, 88.00, 81.00, 49.25, 75.00, 82.86, 85~
## $ samples    <int> 8, 9, 8, 8, 2, 8, 6, 7, 8, 4, 4, 4, 4, 6, 4, 8, 10, 10, 10,~
## $ nondetects <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,~
## $ min        <dbl> 32.00, 31.00, 85.00, 75.00, 81.00, 26.00, 70.00, 70.00, 80.~
## $ max        <dbl> 39.00, 46.00, 120.00, 94.00, 81.00, 68.00, 80.00, 90.00, 90~
```
  






Using historical prior information about TTHM from the municipality, we will adopt a normal-gamma prior distribution,
$\textsf{NormalGamma}(35, 25, 156.25, 24)$ with
a prior mean of 35 parts per billion, a prior sample
size of 25, an estimate of the variance of 156.25 with degrees of freedom 24.  In Section \@ref(sec:NG-predictive), we will describe how we arrived at these values.

Using the summaries of the data, $\bar{Y} = 55.5$,
variance $s^2 = 540.7$ and sample size of $n = 28$ with the
prior hyper-parameters from above, the posterior hyper-parameters are updated as follows:
\begin{eqnarray*}
n_n & = &  25 +  28 = 53\\
m_n  & = & \frac{28 \times55.5 + 25 \times35}{53} = 45.8  \\
v_n & = & 24 + 28 = 52  \\
s^2_n & = & \frac{(n-1) s^2 + v_0 s^2_0 + n_0 n (m_0 - \bar{Y})^2 /n_n }{v_n}  \\
  & = & \frac{1}{52}
     \left[27 \times 540.7 +
          24 \times 156.25  +
          \frac{25 \times 28}{53} \times (35 - 55.5)^2
\right] = 459.9  \\
\end{eqnarray*}
in the conjugate $\textsf{NormalGamma}(45.8, 53, 459.9, 52)$ 
posterior distribution that now summarizes our 
uncertainty about $\mu$ and $\phi$ ($\sigma^2$)  after seeing the data.

We can obtain the updated hyper-parameters in `R` using the following code in `R`

```r
# prior hyper-parameters
m_0 = 35; n_0 = 25;  s2_0 = 156.25; v_0 = n_0 - 1
# sample summaries
Y = tapwater$tthm
ybar = mean(Y)
s2 = var(Y)
n = length(Y)
# posterior hyperparameters
n_n = n_0 + n
m_n = (n*ybar + n_0*m_0)/n_n
v_n = v_0 + n
s2_n = ((n-1)*s2 + v_0*s2_0 + n_0*n*(m_0 - ybar)^2/n_n)/v_n
```




Using the following code in `R` the  95\%
credible interval for the tap water data may be obtained using the Student $t$ quantile function `qt`.


```r
m_n + qt(c(0.025, 0.975), v_n)*sqrt(s2_n/n_n)
```

```
## [1] 39.93192 51.75374
```
The `qt` function takes two arguments:  the first is the desired quantiles, while the second is the degrees of freedom. Both arguments may be vectors, in which case, the result will be a vector.   

While we can calculate the interval directly as above, we have provided the `bayes_inference` function in the `statsr` package to calculate the posterior hyper-parameters, credible intervals and plot  the posterior density and the HPD interval given the raw data:


```r
bayes_inference(tthm, data=tapwater, prior="NG",
                mu_0 = m_0, n_0=n_0, s_0 = sqrt(s2_0), v_0 = v_0,
                stat="mean", type="ci", method="theoretical", 
                show_res=TRUE, show_summ=TRUE, show_plot=FALSE)
```

```
## Single numerical variable
## n = 28, y-bar = 55.5239, s = 23.254
## (Assuming proper prior:  mu | sigma^2 ~ N(35, *sigma^2/25)
## (Assuming proper prior: 1/sigma^2 ~ G(24/2,156.25*24/2)
## 
## Joint Posterior Distribution for mu and 1/sigma^2:
##  N(45.8428, sigma^2/53) G(52/2, 8.6769*52/2)
## 
## Marginal Posterior for mu:
## Student t with posterior mean = 45.8428, posterior scale = 2.9457 on 52 df
## 
## 95% CI: (39.9319 , 51.7537)
```
  
Let's try to understand the arguments to the function. The first argument of the function is the variable of interest, `tthm`, while the second argument is a dataframe with the variable.  The argument `prior="NG"` indicates that we are using a normal-gamma prior; later we will present alternative priors.  The next two lines provide our prior hyper-parameters.  The line with  `stat="mean", type="ci"` indicate that we are interested in inference about the population mean $\mu$ and to calculate a credible interval for $\mu$. The argument `method = theoretical` indicates that we will use the exact quantiles of the Student $t$ distribution to obtain our posterior credible intervals.  Looking at the output the credible interval agrees with the interval we calculated from the summaries using the t quantiles.  The other arguments are logical variables to toggle on/off the various output.  In this case we have suppressed producing the plot of the posterior distribution using the option `show_plot=FALSE`, however, setting this to `TRUE` produces the density and credible interval shown in Figure \@ref{fig:tapwater-post-mu}.

How do we interpret these results? Based on the updated posterior, we find that there is a 95% chance that
the mean TTHM concentration is between 39.9
parts per billion and 51.8 parts per billion, suggesting that for this period that the municipality is in compliance with the limits.
  

```r
ggplot(data=tapwater, aes(x=tthm)) + geom_histogram()
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](04-normalgamma-01-inference_files/figure-latex/unnamed-chunk-2-1.pdf)<!-- --> 


### Section Summary

The normal-gamma conjugate prior for
inference about an unknown mean and variance for samples from a normal
distribution allows simple expressions for updating prior beliefs given the data.   The joint normal-gamma distribution leads to the
Student $t$ distribution for inference about $\mu$ when $\sigma^2$ is unknown.  The Student $t$ distribution can be used to provide 
credible intervals for $\mu$  using `R` or other software that provides quantiles of a standard $t$ distribution.

For the energetic learner who is comfortable with calculus, the  optional material at the end of this section provides more details on how the posterior distributions were obtained and other results in this section.

For those that are ready to move on, we will introduce Monte Carlo sampling  in the next section; Monte Carlo sampling is a simulation method that will allow us to approximate distributions of transformations of the parameters without using calculus or change of variables, as well as assist exploratory data analysis of the prior or posterior distributions.

### (Optional) Derivations 

From Bayes Theorem we have that the joint posterior distribution is proportional to the likelihood of the parameters times the joint prior distribution
\begin{equation}
p(\mu, \sigma^2 \mid y_1, \ldots, y_n)  \propto {\cal L}(\mu, \sigma^2) p(\mu, \sigma^2).
\end{equation} where the
likelihood function for $\mu$ and $\sigma^2$  is proportional to
$$
\begin{aligned}
{\cal L}(\mu, \sigma^2) \propto 
(\sigma^2)^{-n/2}  \times \exp{ \left\{   
-\frac{1}{2} \frac{\sum_{i = 1}^n(y_i - \bar{y})^2 }{\sigma^2}
\right\}}  & \times 
\exp{ \left\{   
-\frac{1}{2} \frac{n (\bar{y} - \mu)^2 }{\sigma^2} \right\}} \\
  \text{function of $\sigma^2$ and data} & \times
  \text{function of $\mu$, $\sigma^2$ and data}
\end{aligned}
$$ 
which depends on the data only through the sum of squares $\sum_{i = 1}^n(y_i - \bar{y})^2$ (or equivalently the sample variance $s^2 = \sum_{i = 1}^n(y_i - \bar{y})^2/(n-1)$) and the sample mean $\bar{y}$.
Since the likelihood function for $(\mu, \phi)$ is obtained by just substituting $1/\phi$ for $\sigma^2$, the likelihood may be re-expressed as
\begin{equation}
{\cal L}(\mu, \phi) \propto 
\phi^{n/2}  \times \exp{ \left\{   
-\frac{1}{2} \phi (n-1) s^2
\right\}}   \times 
\exp{ \left\{   
-\frac{1}{2} \phi n (\bar{y} - \mu)^2  \right\}}.
\end{equation}

This likelihood may be obtained also be obtained by using the sampling distribution for the summary statistics, where
$$\bar{Y}  \mid \mu, \phi \sim \textsf{Normal}(\mu, 1/(\phi n))$$
and is independent of the sample variance (conditional on $\phi$) and has a gamma distribution
$$ 
s^2 \mid \phi \sim  \textsf{Gamma}\left(\frac{n - 1}{2},  \frac{(n-1) \phi}{2}\right)
$$
with degrees of freedom $n-1$ and rate $(n-1) \phi/2$; the likelihood is the product of the two sampling distributions: ${\cal{L}}(\mu, \phi) \propto p(s^2 \mid \phi) p(\bar{Y} \mid \phi)$. 

Bayes theorem in proportional form leads to the joint posterior distribution
$$\begin{aligned}
p(\mu, \phi \mid \data)  \propto & {\cal{L}}(\mu, \phi) p(\phi) p(\mu \mid \phi)  \\
 =  & \phi^{(n-1)/2)} 
 \exp\left\{ - \frac{ \phi (n-1) s^2 }{2}\right\} 
 \text{  (sampling distribution for  $\phi$) }\\
& \times  (n\phi)^{1/2}   \exp\left\{- \frac 1 2  n \phi (\bar{y} - \mu)^2 \right\} \text{  ( sampling distribution for $\mu$)}
\\
& \times \phi^{\nu_0/2 -1} \exp\{- \frac{ \phi \nu_0 s^2_0}{2}\} \text{ (prior for  $\phi$)}
\\
& \times 
(n_0\phi)^{1/2} \frac{1}{\sqrt{(2 \pi)}} \exp\left\{- \frac 1 2  n_0 \phi (\mu - m_0)^2 \right\} \text{ (prior for $\mu$)}
\end{aligned}
$$
where we have ignored constants that do not involve $\phi$ or $\mu$.
Focusing on all the terms that involve $\mu$, we can group the lines corresponding to the sampling distribution and prior for $\mu$  together and using the factorization of likelihood and prior distributions, we may identify that 
$$p(\mu \mid \phi, \data)  \propto  \exp\left\{- \frac 1 2  n \phi (\bar{y} - \mu)^2  - \frac 1 2  n_0 \phi (\mu - m_0)^2 \right\} 
$$
where the above expression includes the sum of two quadratic expressions in the exponential.   This almost looks like a normal. Can these be combined to form one quadratic expression that looks like a normal density?  Yes!   This is known as "completing the square".
Taking a normal distribution for a parameter $\mu$ with mean $m$ and precision $\rho$, the quadratic term in the exponential may be expanded as
$$\rho \times (\mu - m)^2 = \rho  \mu^2 - 2 \rho  \mu m + \rho  m^2.$$
From this  we can read off that the precision is the term that multiplies the quadratic in $\mu$ and the term that multiplies the linear term in $\mu$ is the product of  two times the mean and precision; this means that if we know the precision, we can identify the mean.  The last term is the precision times the mean squared, which we will need to fill in once we identify the precision and mean.

For our posterior,  we need to expand  the quadratics and recombine
terms to identify the new precision (the coefficient multiplying the quadratic in $\mu$) and the new mean (the linear term) and complete the square so that it may be factored.  Any left over terms will be independent of $\mu$ but may depend on $\phi$.  For our case, after some algebra to group terms we have
\begin{align*}
- \frac 1 2 \left( n \phi (\bar{y} - \mu)^2  +  n_0 \phi (\mu - m_0)^2 \right) & = 
-\frac 1 2 \left(\phi( n + n_0) \mu^2 - 2 \phi \mu (n \bar{y} + n_0 m_0) + \phi (n \bar{y}^2 + n_0 m_0^2) \right)  
\end{align*}
where we can read off that the posterior precision is $\phi(n + n_0) \equiv \phi n_n$.   The linear term is not yet of the form of the posterior precision times the posterior mean (times 2), but if we multiply and divide by $n_n = n + n_0$ it is in the appropriate form
 \begin{equation}-\frac 1 2 \left(\phi( n + n_0) \mu^2 - 2 \phi ( n + n_0) \mu \frac{(n \bar{y} + n_0 m_0) } {n + n_0} + \phi (n \bar{y}^2 + n_0 m_0^2) \right) \label{eq:quad}
 \end{equation}
 so that we may identify that the posterior mean is $m_n = (n \bar{y} + n_0 m_0) /(n + n_0)$ which combined with the precision (or inverse variance) is enough to identity the conditional posterior distribution for $\mu$.
We next add the precision times the square of the posterior mean  (the completing the square part), but to keep equality, we will need to subtract the term as well:
$$
- \frac 1 2 \left( n \phi (\bar{y} - \mu)^2  +  n_0 \phi (\mu - m_0)^2 \right)  = 
-\frac 1 2 \left(\phi n_n \mu^2 - 2 \phi n_n \mu m_n + \phi n_n m_n^2  - \phi n_n m_n^2  + \phi (n \bar{y}^2 + n_0 m_0^2) \right)  
$$
which after factoring the quadratic leads to 
\begin{align}
 - \frac 1 2 \left( n \phi (\bar{y} - \mu)^2  +  n_0 \phi (\mu - m_0)^2 \right) = & -\frac 1 2 \left(\phi n_n (\mu - m_n)^2 \right) \\
 & -\frac 1 2 \left(\phi (-n_n m_n^2 + n \bar{y}^2 + n_0 m_0^2) \right) 
\end{align}
where the  first line is the quadratic for the posterior of $\mu$ given $\phi$ while the second line includes terms that involve $\phi$ but that are independent of $\mu$.

Substituting the expressions, we can continue to simplify the expressions further 
$$
\begin{aligned}
p(\mu, \phi \mid \data)  \propto & {\cal{L}}(\mu, \phi) p(\phi) p(\mu \mid \phi)  \\
 =  & \phi^{(n + \nu_0 + 1  )/2 - 1} 
 \exp\left\{ - \frac{ \phi (n-1) s^2 }{2}\right\} 
\times \exp\left\{- \frac{ \phi \nu_0 s^2_0}{2}\right\} \times
\exp\left\{  -\frac 1 2 \left(\phi (-n_n m_n^2 + n \bar{y}^2 + n_0 m_0^2) \right)   \right\} 
\\
& \times \exp \left\{ -\frac 1 2 \left(\phi n_n (\mu - m_n)^2 \right) \right\} \\
=   & \phi^{(n + \nu_0)/2 - 1} 
 \exp\left\{ -  \frac{\phi}{2}  \left( (n-1) s^2  + \nu_0 s^2_0 +
\frac{ n_0 n }{n_n} ( m_0 - \bar{y})^2 \right)   \right\}   \qquad \qquad
\text{ (gamma kernel)} \\
& \times (n_n \phi)^{1/2} \exp \left\{ -\frac 1 2 \left(\phi n_n (\mu - m_n)^2 \right) \right\} \qquad \qquad \text{ (normal kernel})
\end{aligned}
$$
until we can recognize the product of the kernels of  a gamma distribution for $\phi$
$$
\phi \mid \data \sim \St(v_n/2, v_n s^2_n/ 2)
$$
where $\nu_n = n + \nu_0$ and $s^2_n = \left((n-1) s^2 + n_0 s^2_0 + (m_0 - \bar{y})^2 n n_0/n_n\right)/\nu_n$ times a normal:
and the kernel of a normal for $\mu$ 
$$ \mu \mid \phi, \data \sim \No(m_n, (\phi n_n)^{-1})
$$
where $m_m = (n \bar{y} + n_0 m_0) /(n + n_0)$ a weighted average of the sample mean and the prior mean, and
$n_n = n + n_0$ is the sample and prior combined sample size.





#### Derivation of Marginal Distribution for $\mu$


If $\mu$ given $\sigma^2$ (and the data) has a normal distribution with mean $m_m$ and variance $\sigma^2/n_n$ and $1/\sigma^2 \equiv \phi$ (given the data) has a gamma distribution with shape parameter 
$\nu_n/2$ and rate parameter $\nu_n s^2_n/2$ 
  
$$\begin{aligned}
\mu \mid \sigma^2, \data & \sim \No(m_m, \sigma^2/n_n) \\ 
1/\sigma^2 \mid \data & \sim \Ga(\nu_n/2, \nu_n s^2_n/2) 
\end{aligned}$$
then
$$\mu \mid  \data  \sim \St(\nu_n, m_m, s^2_n/n_n)$$
a Student $t$ distribution with mean $m_m$ and scale $s^2_n/n_n$ with degrees of freedom $\nu_n$. 

This applies to the prior as well, so that without any data we use the prior hyper-parameters $m_0$, $n_0$, $\nu_0$ and $s^2_0$ in place of the updated values with the subscript $n$.  

To simplify notation, we'll substitute $\phi = 1/\sigma^2$.  The marginal distribution for $\mu$ is obtained by averaging over the values of $\sigma^2$.  Since $\sigma^2$ takes on continuous values rather than discrete, this averaging is represented as an integral
\begin{align*}
p(\mu \mid \data) & = \int_0^\infty p(\mu \mid \phi, \data) p(\phi \mid \data) d\phi \\
 & = \int_0^\infty 
\frac{1}{\sqrt{2 \pi}} (n_n \phi)^{1/2} 
e^{\left\{ - \frac{n_n \phi}{2} (\mu - m_n)^2 \right\}}
\frac{1}{\Gamma(\nu_n/2)} \left(\frac{\nu_n s^2_n}{2}\right)^{\nu_n/2}
\phi^{\nu_n/2 - 1} e^{\left\{- \phi \nu_n s^2_n/2\right\}} \, d\phi \\
 & =  
\left(\frac{n_n}{2 \pi}\right)^{1/2}\frac{1}{\Gamma\left(\frac{\nu_n}{2}\right)} \left(\frac{\nu_n s^2_n}{2}\right)^{\nu_n/2} \int_0^\infty  \phi^{(\nu_n +1)/2 - 1}
e^{\left\{ - \phi \left( \frac{n_n  (\mu - m_n)^2 + \nu_n s^2_n}{2} \right)\right\}} \, d\phi
\intertext{where the terms inside the integral are the "kernel" of a Gamma density.  We can multiply and divide by the normalizing constant of the Gamma density}
p(\mu \mid \data) & =  
\left(\frac{n_n}{2 \pi}\right)^{1/2}\frac{1}{\Gamma\left(\frac{\nu_n}{2}\right)} 
\left(\frac{\nu_n s^2_n}{2}\right)^{\nu_n/2} 
\Gamma\left(\frac{\nu_n + 1}{2}\right) 
\left( \frac{n_n  (\mu - m_n)^2 + \nu_n s^2_n}{2} \right)^{- \frac{\nu_n + 1}{2}}  \times\\
& \qquad \int_0^\infty  \frac{1}{\Gamma\left(\frac{\nu_n + 1}{2}\right)}
\left( \frac{n_n  (\mu - m_n)^2 + \nu_n s^2_n}{2} \right)^{ \frac{\nu_n + 1}{2}} \phi^{(\nu_n +1)/2 - 1}
e^{\left\{ - \phi \left( \frac{n_n  (\mu - m_n)^2 + \nu_n s^2_n}{2} \right)\right\}} \, d\phi
\intertext{so that the term in the integral now integrates to one and the resulting distribution is}
p(\mu \mid \data) & =  
\left(\frac{n_n}{2 \pi}\right)^{1/2}\frac{\Gamma\left(\frac{\nu_n + 1}{2}\right) }{\Gamma\left(\frac{\nu_n}{2}\right)} 
\left(\frac{\nu_n s^2_n}{2}\right)^{\nu_n/2} 
\left( \frac{n_n  (\mu - m_n)^2 + \nu_n s^2_n}{2} \right)^{- \frac{\nu_n + 1}{2}}.
\intertext{After some algebra this simplifies to}
p(\mu \mid \data) & =  
\frac{1}{\sqrt{\pi \nu_n s^2_n/n_n}}
\frac{\Gamma\left(\frac{\nu_n + 1}{2}\right) }
     {\Gamma\left(\frac{\nu_n}{2}\right)} 
\left( 1 +  \frac{1}{\nu_n}\frac{(\mu - m_n)^2}{s^2_n/n_n} \right)^{- \frac{\nu_n + 1}{2}}
\intertext{and is a more standard representation for a Student $t$ distribution and the  kernel of the density is the right most term.}
\end{align*}
