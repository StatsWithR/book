## The Normal-Gamma Conjugate Family {#sec:normal-gamma}


You may take the safety of your drinking water for granted, however, residents of Flint,  Michigan were outraged over reports that the levels of a contaminant known as __TTHM__ exceeded federal allowance levels in 2014.   TTHM stands for  total trihalomethanes, a group of chemical compounds first identified in drinking water in the 1970’s that form during drinking water treatment when organic matter in natural water reacts chemically with chlorine disinfectants. Since violations are based on annual running averages, we are interested in inference about the mean TTHM level based on measurements taken from samples.  

In Section \@ref(sec:normal-normal) we described the normal-normal conjugate  family for inference about an unknown mean $\mu$ when the data $Y_1, Y_2, \ldots, Y_n$ were assumed to be a random sample of size $n$ from a normal population  with a known standard deviation $\sigma$, however, it is more common in practice to have data where the variability of obaservations is unknown, as in the example with TTHM. Conceptually, Bayesian inference for two (or more) parameters is not any different from the case with one parameter.  As both $\mu$ and $\sigma^2$ unknown, we will need to specify a **joint** prior distribution, $p(\mu, \sigma^2)$ to describe our prior uncertainty about them. As before, Bayes Theorem leads the posterior distribution for $\mu$ and $\sigma^2$ given the observed data

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


As in the earlier chapters, conjugate priors are appealing as there are nice expressions for updating the prior to obtain the posterior distribution. In the case of two parameters or more parameters a conjugate pair is a sampling model for the data  and a joint prior distribution for the unknown parameters such that the joint posterior distribution is in the same family of distributions as the prior distribution.  In this case our sampling model is built on the assumption that the data are a random sample of size $n$ from a normal population with mean $\mu$ and variance $\sigma^2$, expressed in  shorthand as

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
as the product of a **conditional distribution** for $\mu$ given $\sigma^2$ and a **marginal distribution** for $\sigma^2$, suggests that the posterior distribution should factor as the product of two conjugate distributions. Perhaps not surprisingly, this is indeed the case.

### Conjugate Prior for $\mu$ and $\sigma^2$

In Section \@ref(sec:normal-normal), we found that for normal data, the conjugate prior distribution for $\mu$ when the standard deviation $\sigma$ was known was a normal distribution.  We will build on this to specify a conditional prior distribution for $\mu$ as a normal distribution
\begin{equation}
\mu \mid \sigma^2   \sim  \textsf{N}(m_0, \sigma^2/n_0)
(\#eq:04-conjugate-normal)
\end{equation}
with hyper-parameters $m_0$, the prior mean for $\mu$, and $\sigma^2/n_0$ the prior variance.  While previously we represented the prior variance as a fixed constant, $\tau^2$, in this case we will replace $\tau^2$ with a multiple of $\sigma^2$.    Because $\sigma$ has the same units as the data, the presence of $\sigma$ in the prior variance automatically scales the prior for $\mu$ based on the same units.  This is important, for example, if we were to change the measurement units from inches to centimeters or seconds to hours, as the prior will be re-scaled automatically. The hyper-parameter $n_0$ is unitless, but is used to express our prior precision about $\mu$ relative to the level of "noise", captured by $\sigma^2$, in the data.  Larger values of $n_0$ indicate that we know the mean with more precision (relative to the variabiltiy in observations) with smaller values indicating less precision or more uncertainty.  We will see later how the hyper-parameter $n_0$ may be interpreted as a prior sample size.  Finally, while we could use a fixed value $\tau^2$ as the prior variance in a conditional conjugate prior for $\mu$ given $\sigma^2$,  that does not lead to a joint conjugate prior for $\mu$ and $\sigma^2$.

As $\sigma^2$ is unknown, a Bayesian would use a
prior distribution to describe the uncertainty about the variance before seeing data.  Since the variance is non-negative, continuous, and with no upper limit,  based on the distributions that we have seen so far a gamma distribution might appear to be a candidate  prior for the variance,. However, that choice does not lead to a posterior distribution in the same family or that is recognizable as any common distribution. It turns out that the the inverse of the variance, which is known as the precision, has a conjugate gamma prior distribution.

For simplification let's express the precision (inverse variance) as a new parameter,  $\phi = 1/\sigma^2$.  Then the conjugate prior for $\phi$,
\begin{equation}
\phi \sim \textsf{Gamma}\left(\frac{v_0}{2}, \frac{v_0 s^2_0}{2} \right)
(\#eq:04-conjugate-gamma)
\end{equation}
is a gamma distribution with  shape parameter $v_0/2$ and  **rate** parameter of ${v_0 s^2_0}/{2}$. Given the connections between the gamma distribution and the Chi-Squared distribution,  the hyper-parameter $v_0$ may be interpreted as the prior degrees of freedom.  The hyperparameter $s^2_0$ may be interpreted as a prior variance or initial prior estimate for $\sigma^2$. Equivalently, we may say that the inverse of the variance has a 
$$1/\sigma^2 \sim \textsf{Gamma}(v_0/2, s^2_0 v_0/2)$$

gamma distribution to avoid using a new symbol \footnote{In some other references, you will see that $\sigma^2$ will have an inverse gamma distribution.  Rather than introduce an additional distribution for the inverse-gamma, we will restrict our attention to the gamma distribution since the inverse-gamma is equivalent to saying that the inverse of $\sigma^2$ has a gamma distribution and `R` has support for generating random variables from the gamma that we will need in later sections.}. Together the conditional normal distribution for $\mu$ given $\sigma^2$ in \@ref(eq:04-conjugate-normal)  and the marginal gamma distribution for $\phi$ in \@ref(eq:04-conjugate-gamma) lead to a joint distribution for the pair $(\mu, \phi)$ that we will call the normal-gamma family of distributions:
\begin{equation}(\mu, \phi) \sim \textsf{NormalGamma}(m_0, n_0, s^2_0, v_0)
(\#eq:04-conjugate-normal-gamma)
\end{equation}
with the four hyper-parameters $m_0$, $n_0$, $s^2_0$, and $v_0$.

These have simple rules for updating given new data to obtain the posterior hyperparmeters due to conjugacy.

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

The *standardized Student $t$-distribution*  with $\nu$ degrees of freedom is defined to be
$$ p(t) = \frac{1}{\sqrt{\pi\nu}}\frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})}\left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}. $$ 
The $\Gamma(\cdot)$ is the Gamma function that we have seen earlier. This Student's $t$-distribution is centered at 0 (the location parameter), with a scale parameter equal to 1, like in a standard normal, however, there is an additional parameter, $\nu$, the degrees of freedom parameter.

The Student $t$ distribution is similar to the normal distribution as it is symmetric about the center and bell shaped, however, the __tails__ of the distribution are fatter or heavier than the normal distribution   and therefore, it is a little "shorter" in the middle as illustrated in Figure \@ref(fig:density)


![(\#fig:t-density)Standard normal and Student t densities.](04-normalgamma-01-inference_files/figure-latex/t-density-1.pdf) 

Similar to the normal distribution, we can obtain other Student $t$ distributions  by changing the center of the distribution and changing the scale.

We are now ready for our main result.  
\BeginKnitrBlock{definition}<div class="definition"><span class="definition" id="def:unnamed-chunk-1"><strong>(\#def:unnamed-chunk-1) </strong></span>If $\mu$ and $1/\sigma^2$ have a $\textsf{NormalGamma}(m_n, n_n, v_n, s^2_n)$ posterior distribution, then 
$\mu$ given the data has a \index{Student t distribution} distribution, $\St(v_n, m_n, s^2_n/n_n)$, expressed as
$$ \mu \mid \data \sim \St(v_n, m_n, s^2_n/n_n)  $$ 
with degrees of freedom $v_n$, 
location parameter, $m_n$, and squared scale parameter, $s^2_n/n_n$, that is the
posterior variance parameter divided by the posterior sample size.  
</div>\EndKnitrBlock{definition}

The parameters $m_n$ and $s^2_n$ play similar roles in determining the center and spread of the distribution, as in the normal distribution, however,  as Student $t$ distributions with degrees of freedom less than 3 do not have a mean or variance, the parameter $m_n$ is called the location or center of the distribution and the $s_n/\sqrt{n}$ is the scale.

The  density for a $\St(v_n, m_m, s^2_n/n_n)$ random variable is
\begin{equation}
p(\mu) =\frac{\Gamma\left(\frac{v_n + 1}{2} \right)}
{\sqrt{\pi v_n} \frac{s_n}{\sqrt{n_n}} \,\Gamma\left(\frac{v_n}{2} \right)}
\left(1 + \frac{1}{v_n}\frac{(\mu - m_n)^2} {s^2_n/n_n} \right)^{-\frac{v_n+1}{2}} 
(\#eq:Student-t-density)
\end{equation}
and by subtracting the location $m_n$ and dividing by the scale 
$s_n/\sqrt{n}$:
$$ \frac{\mu - m_n}{s_n/\sqrt{n_n}} \equiv t \sim \St(v_n, 0 , 1)  $$
we can obtain the distribution of the standardized Student $t$ distribution with degrees of freedom $v_n$, location  $0$ and scale $1$.
This latter representation allows us to use standard statistical functions for posterior inference such as finding credible intervals.




\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:example-tapwater"><strong>(\#exm:example-tapwater) </strong></span>**TTHM in Tapwater**  A municipality in North Carolina is interested the levels of TTHM in their drinking water.   The data can be loaded from the `statsr` package in `R`.</div>\EndKnitrBlock{example}


  
  ```r
  library(statsr)
  data(tapwater)
  str(tapwater)
  ```
  
  ```
  ## 'data.frame':	28 obs. of  6 variables:
  ##  $ date      : Factor w/ 28 levels "2004-02-19","2004-03-22",..: 28 27 26 25 24 23 22 21 20 19 ...
  ##  $ tthm      : num  34.4 39.3 108.6 88 81 ...
  ##  $ samples   : int  8 9 8 8 2 8 6 7 8 4 ...
  ##  $ nondetects: int  0 0 0 0 0 0 0 0 0 0 ...
  ##  $ min       : num  32 31 85 75 81 26 70 70 80 82 ...
  ##  $ max       : num  39 46 120 94 81 68 80 90 90 92 ...
  ```
  


  




Using prior information about TTHM from the city, we will use a normal-gamma prior distribution,
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
  # prior hyperparameters
  m_0 = 35; n_0 = 25;  s2_0 = 156.25; v_0 = n_0 - 1
  # sample summaries
  Y = tapwater$tthm
  ybar = mean(Y)
  s2 = var(Y)
  n = length(Y)
  # posterior hyperparamters
  n_n = n_0 + n
  m_n = (n*ybar + n_0*m_0)/n_n
  v_n = v_0 + n
  s2_n = ((n-1)*s2 + v_0*s2_0 + n_0*n*(m_0 - ybar)^2/n_n)/v_n
  ```



### Credible Intervals for $\mu$

To find a credible interval for the mean $\mu$, we use the Student $t$
distribution. 


\begin{figure}

{\centering \includegraphics{04-normalgamma-01-inference_files/figure-latex/tapwater-post-mu-1} 

}

\caption{Highest Posterior Density region for the mean in the total trihalomethanes tapwater example.}(\#fig:tapwater-post-mu)
\end{figure}

Since the distribution of $\mu$ is unimodal and symmetric, the shortest 95 percent credible interval or the **Highest Posterior Density** interval, HPD for short,
is the orange interval given by the
Lower endpoint L and upper endpoint U where the probability that $\mu$ is
in the interval (L, U) is the shaded area which is equal to zero point
nine five.

Using the standardized Student $t$ distribution and some algebra, these values are
$$
\begin{aligned}
  L & =  m_n + t_{0.025}\sqrt{s^2_n/n_n}    \\
  U & =  m_n + t_{0.975}\sqrt{s^2_n/n_n}
\end{aligned}
$$
or the posterior mean (our point estimate) plus quantiles of the standard $t$ distribution times the scale.  Because of the symmetry in the Student $t$ distribution, the credible interval is $m_n \pm t_{0.975}\sqrt{s^2_n/n_n}$, which should look familiar to expressions for confidence intervals.

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:tapwater-CI"><strong>(\#exm:tapwater-CI) </strong></span>**TTHM in Tapwater (continued)**
Using the following code in `R` the  95\%
credible interval for the tap water data is
</div>\EndKnitrBlock{example}
  
  ```r
  m_n + qt(c(0.025, 0.975), v_n)*sqrt(s2_n/n_n)
  ```
  
  ```
  ## [1] 39.93192 51.75374
  ```

Based on the updated posterior, we find that there is a 95% chance that
the mean TTHM concentration is between 39.9
parts per billion and 51.7 parts per billion.
  


### Summary

The normal-gamma conjugate prior for
inference about an unknown mean and variance for samples from a normal
distribution allows simple expressions for updating prior beliefs given the data.   The joint normal-gamma distribution leads to the
Student $t$ distribution for inference about $\mu$ when $\sigma^2$ is unknown.  The Student $t$ distribution can be used to provide 
credible intervals for $\mu$  using `R` or other software that provides quantiles of a standard $t$ distribution.

For the energetic learner who is comfortable with calculus, the  optional material at the end of this section provides more details on how the posterior distributions were obtained and other results in this section.

For those that are ready to move on, we will introduce Monte Carlo sampling  in the next section; Monte Carlo Sampling is a simulation method that will allow us to approximate distributions of transformations of the parameters without using calculus or change of variables, as well as aid exploratory data analysis of the prior or posterior distribution.

### (Optional) Derivations 

From Bayes Theorem we have that 
\begin{equation}
p(\mu, \sigma^2 \mid y_1, \ldots, y_n)  \propto {\cal L}(\mu, \sigma^2) p(\mu, \sigma^2).
\end{equation}
the posterior distribution is proportional to the likelihood of the parameters times the prior distriubtion where the
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


We could have started with the sampling distribution for these statistics, where

$$\bar{Y}  \mid \mu, \sigma^2 \sim \textsf{Normal}(\mu, \sigma^2/n)$$
and is independent of the sample variance,
$$ 
s^2 \mid \sigma^2 \sim  \textsf{Gamma}\left(\frac{n - 1}{2},  \frac{n-1}{2 \sigma^2}\right)
$$
with degrees of freedom $\nu = n-1$ and rate $(n-1)/(2 \sigma^2)$.
