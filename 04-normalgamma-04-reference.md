## Reference Priors  {#sec:NG-reference}



In Section \@ref(sec:NG-predictive), we described a way of specifying an informative prior distribution for inference about TTHM in tapwater based on additional prior information. We had to use a prior sample size that was comparable to the observed sample size for the prior predictive under the Normal-Gamma distribution to agree with the reported prior interval.

However, there may be cases where prior information is not available, or you may wish to present an objective analysis where minimal prior information is used. Or perhaps, you want to use the Bayesian paradigm to make probability statements about parameters, but not use any prior information.

In this section, we will present reference priors for normal data, which can be viewed as a limiting form of the Normal-Gamma conjugate prior distribution. **Can you actually perform a Bayesian analysis without using prior information?**

Conjugate priors can be interpreted to be based on a prior sample. What happens in the conjugate Normal-Gamma prior if we take our prior sample size $n_0$ to go to zero?  If we have no data, then we will define the prior sample variance $s_0^2$ to go to  0, and based on the relationship between prior sample sized and prior degrees of freedom, we will let the prior degrees of freedom go to the prior sample size minus one, or negative one, i.e. $v_0 = n_0 - 1 \rightarrow -1$.

With this limit, we have the following properties:

* The posterior mean goes to the sample mean.

* The posterior sample size is the observed sample size.

* The posterior degrees of freedom go to the sample degrees of freedom.

* The posterior variance parameter goes to the sample variance.

In this limit, the posterior hyperparameters do not depend on the prior hyperparameters.

Since $n_0 \rightarrow 0, s^2_0 \rightarrow 0, v_0 = n_0 - 1 \rightarrow -1$, we have in mathematical terms:

$$\begin{aligned}
m_n &= \frac{n \bar{Y} + n_0 m_0} {n + n_0}  \rightarrow \bar{Y} \\
n_n &= n_0 + n  \rightarrow n \\
v_n &= v_0 + n  \rightarrow n-1 \\
s^2_n &= \frac{1}{v_n}\left[s^2_0 v_0 + s^2 (n-1) + \frac{n_0 n}{n_n} (\bar{Y} - m_0)^2 \right] \rightarrow s^2
\end{aligned}$$

This limiting normal-gamma distribution, $\textsf{NormalGamma}(0,0,0,-1)$, is not really a normal-gamma distribution, as the density does not integrate to 1. The form of the limit can be viewed as a prior for $\mu$ that is proportional to a constant, or uniform/flat on the whole real line. And a prior for the variance is proportional to 1 over the variance. The joint prior is taken as the product of the two.


$$\begin{aligned}
p(\mu \mid \sigma^2) & \propto  1 \\
p(\sigma^2) & \propto  1/\sigma^2 \\
p(\mu, \sigma^2) & \propto  1/\sigma^2
\end{aligned}$$

This is refered to as a **reference prior** because the posterior hyperparameters do not depend on the prior hyperparameters.

In addition, $\textsf{NormalGamma}(0,0,0,-1)$ is a special case of a reference prior, known as the independent Jeffreys prior.  While Jeffreys used other arguments to arrive at the form of the prior, the goal was to have an **objective prior** invariant to shifting the data by a constant or multiplying by a constant.

Now, a naive approach to constructing a non-informative distribution might be to use a uniform distribution to represent lack of knowledge. However, would you use a uniform distribution for $\sigma^2$, or a uniform distribution for the precision $1/\sigma^2$? Or perhaps a uniform distribution for $\sigma$? These would all lead to different posteriors with little justification for any of them. This ambiguity led Sir Harold Jeffreys to propose reference distributions for the mean and variance for situations where prior information was limited. These priors are **invariant** to the units of the data.

The unnormalized priors that do not integrate to a constant are called **improper distributions**. An important consideration in using them is that one cannot generate samples from the prior or the prior predictive distribution to data and are referred to as **non-generative distributions**.

While the reference prior is not a proper prior distribution, and cannot reflect anyone's actual prior beliefs, the formal application phase rule can still be used to show that **the posterior distribution is a valid normal gamma distribution**, leading to a formal phase posterior distribution. That depends only on summary statistics of the data.

The posterior distribution $\textsf{NormalGamma}(\bar{Y}, n, s^2, n-1)$ breaks down to

$$\begin{aligned}
\mu \mid \sigma^2, \data & \sim \No(\bar{Y}, \sigma^2/n) \\
1/\sigma^2  \mid \data & \sim \Ga((n-1)/2, s^2(n - 1)/2).
\end{aligned}$$

* Under the reference prior $p(\mu, \sigma^2) \propto 1/\sigma^2$, the posterior distribution  after standardizing $\mu$ has a Student $t$  distribution with $n-1$ degrees of freedom.

$$\frac{\mu - \bar{Y}}{\sqrt{s^2/n}} \mid \data \sim  \St(n-1, 0, 1)$$

* Prior to seeing the data, the distribution of the standardized sample mean given $\mu$ and $\sigma$ also has a Student t distribution.

$$\frac{\mu - \bar{Y}}{\sqrt{s^2/n}} \mid \mu, \sigma^2 \sim  \St(n-1, 0, 1) $$

* Both frequentist sampling distributions and Bayesian reference posterior distributions lead to intervals of this form:

$$(\bar{Y} - t_{1 - \alpha/2}\times s/\sqrt{n}, \, \bar{Y} + t_{1 - \alpha/2} \times s/\sqrt{n})$$

* However, only the Bayesian approach justifies the probability statements about $\mu$ being in the interval after seeing the data.

$$P(\bar{Y} - t_{1 - \alpha/2}\times s/\sqrt{n} < \mu <  \bar{Y} + t_{1 - \alpha/2}\times s/\sqrt{n}) = 1 - \alpha$$

We can use either analytic expressions based on the t-distribution, or Monte Carlo samples from the posterior predictive distribution, to make predictions about a new sample.



Here is some code to generate the Monte Carlo samples from the tap water example:

```r
phi = rgamma(10000, (n-1)/2, s2*(n-1)/2)
sigma = 1/sqrt(phi)
post_mu = rnorm(10000, mean=ybar, sd=sigma/(sqrt(n)))
pred_y =  rnorm(10000,post_mu, sigma)
quantile(pred_y, c(.025, .975))
```

```
##       2.5%      97.5% 
##   6.692877 104.225954
```

Using the Monte Carlo samples, Figure \@ref(fig:plot-post-pred) shows the posterior distribution based on the informative Normal-Gamma prior and the reference prior. Both the posterior distribution for $\mu$ and the posterior predictive distribution for a new sample are shifted to the right, and are centered at the sample mean. The posterior for $\mu$ under the reference prior is less concentrated around its mean than the posterior under the informative prior, which leads to an increased posterior sample size and hence increased precision.

\begin{figure}

{\centering \includegraphics{04-normalgamma-04-reference_files/figure-latex/plot-post-pred-1} 

}

\caption{Comparison of posterior densities}(\#fig:plot-post-pred)
\end{figure}

The posterior probability that a new sample will exceed the legal limit of 80 ppb under the reference prior is roughly 0.15, which is more than double the probability of 0.06 from the posterior under the informative prior.


```r
sum(pred_y > 80)/length(pred_y)  # P(Y > 80 | data)
```

```
## [1] 0.1534
```

In constructing the informative prior from the reported interval, there are two critical assumptions. First, the prior data are exchangeable with the observed data. Second, the conjugate normal gamma distribution is suitable for representing the prior information. These assumptions may or may not be verifiable, but they should be considered carefully when using informative conjugate priors.

In the case of the tap water example, there are several concerns: One, it is unclear that the prior data are exchangeable with the observed data.  For example, water treatment conditions may have changed. Two, the prior sample size was not based on a real prior sample, but instead selected so that the prior predictive intervals under the normal gamma model agreed with the prior data.  As we do not have access to the prior data, we cannot check assumptions about normality that would help justify the prior. Other skewed distributions may be consistent with the prior interval, but lead to different conclusions.

To recap,  we have introduced a  reference prior for inference for
normal data with an unknown mean and variance. Reference priors are often part of a prior sensitivity study and are used when objectivity is of utmost importance.

If conclusions are fundamentally different with an informative prior and a reference prior, one may wish to carefully examine assumputions that led to the informative prior.

* Is the prior information based on a prior sample that is exchangable with the observed data?

* Is the normal-gamma assumption appropriate?

Informative priors can provide more accurate inference when data are limited, and the transparency of explicitly laying out prior assumptions is an important aspect of reproducible research. However, one needs to be careful that certain prior assumptions may lead to un-intended consequences.

Next, we will investigate a prior distribution that is a mixture of
conjugate priors, so the new prior distribution provides robustness to prior mis-specification in the prior sample size.

While we will no longer have nice analytical expressions for the posterior, we can simulate from the posterior distribution using a Monte Carlo algorithm
called Markov chain Monte Carlo (MCMC).
