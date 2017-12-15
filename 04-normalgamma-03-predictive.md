
### Predictive Distributions {#sec:NG-predictive}


```r
library(statsr)
library(ggplot2)
```





In this section, we will discuss prior and posterior **predictive** distributions of the data and show how Monte Carlo sampling from the prior predictive distribution can help select hyper parameters.

We can obtain the prior predictive distribution of the data, by taking the joint distribution of the data and the parameters in averaging over the possible values of the parameters from the prior.

* Prior:

$$ \begin{aligned}
\frac{1}{\sigma^2} = \phi &\sim \textsf{Gamma}\left(\frac{v_0}{2}, \frac{v_0 s^2_0}{2} \right) \\
\mu \mid \sigma^2  &\sim  \textsf{N}(m_0, \sigma^2/n_0)
\end{aligned} $$

* Sampling model:

$$Y_i \mid \mu,\sigma^2 \iid \No(\mu, \sigma^2) $$

* Prior predictive distribution for $Y$:

$$\begin{aligned}
p(Y) &= \iint p(Y \mid \mu,\sigma^2) p(\mu \mid \sigma^2) p(\sigma^2) d\mu \, d\sigma^2 \\
Y &\sim t(v_0, m_0, s_0^2+s_0^2/n_0)
\end{aligned}$$

This distribution of the observables can be used to help elicit prior hyper parameters as in the tap water example.

A report from the city water department suggests that levels of TTHM are expected to be between 10-60 parts per billion (ppb).

* Set the prior mean $\mu$ to be at the midpoint of the interval: $m_0 = (60+10)/2 = 35$

* Standard deviation: Based on the empirical rule, 95% observations are within $\pm 2\sigma$ of $\mu$, we expect that the range of the data should be $4\sigma$.

* Prior estimate of sigma: $s_0 = (60-10)/4 = 12.5$ or $s_0^2 = [(60-10)/4]^2 = 156.25$

To complete the specification, we also need to choose the prior sample size $n_0$ and degrees of freedom $v_0$. As the degrees of freedom of the variance are $n-1$, we set $v_0 = n_0 - 1$. We will draw samples from the prior predictive distribution and modify $n_0$ so that the simulated data agree with our prior assumptions.

The following `R` code shows a simulation from the predictive distribution with the prior sample size of 2. Please note that the number of Monte Carlo simulations should not be confused with the prior sample size $n_0$.

We begin by simulating $\phi$, transfering $\phi$ to calculate $\sigma$, and then simulating values of $\mu$. Finally, the simulated values of $\mu,\sigma$ are used to generate possible values of TTHM denoted by $Y$.


```r
m_0 = (60+10)/2; s2_0 = ((60-10)/4)^2;
n_0 = 2; v_0 = n_0 - 1
set.seed(1234)
phi = rgamma(10000, v_0/2, s2_0*v_0/2)
sigma = 1/sqrt(phi)
mu = rnorm(10000, mean=m_0, sd=sigma/(sqrt(n_0)))
y = rnorm(10000, mu, sigma)
quantile(y, c(0.025,0.975))
```

```
##      2.5%     97.5% 
## -140.1391  217.7050
```

This forward simulation propagates uncertainty in $\mu,\sigma$ to the prior predictive distribution of the data. Calculating the sample quantiles from the samples of the prior predictive for $Y$, we see that the 95% predictive interval includes negative values. Since TTHM is non-negative, we need to adjust $n_0$ and repeat.

After some trial and error, we find that the prior sample size of 25 (in fact the Central Limit Theorem suggests at least 25 or 30 to be "sufficiently large"), the empirical quantiles from the prior predictive distribution are close to the range of 10 to 16 that we were given as prior information.


```r
m_0 = (60+10)/2; s2_0 = ((60-10)/4)^2;
n_0 = 25; v_0 = n_0 - 1
set.seed(1234)
phi = rgamma(10000, v_0/2, s2_0*v_0/2)
sigma = 1/sqrt(phi)
mu = rnorm(10000, mean=m_0, sd=sigma/(sqrt(n_0)))
y = rnorm(10000, mu, sigma)
quantile(y, c(0.025,0.975))
```

```
##      2.5%     97.5% 
##  8.802515 61.857350
```

Figure \@ref(fig:hist-prior) shows an estimate of the prior distribution of $\mu$ in gray and the more dispersed prior predictive distribution in TTHM in orange, obtained from the Monte Carlo samples.

\begin{figure}

{\centering \includegraphics{04-normalgamma-03-predictive_files/figure-latex/hist-prior-1} 

}

\caption{Prior density}(\#fig:hist-prior)
\end{figure}

Using the Monte Carlo samples, we can also estimate the prior probability of negative values of TTHM by counting the number of times the simulated values are less than zero out of the total number of simulations.


```r
sum(y < 0)/length(y)  # P(Y < 0) a priori
```

```
## [1] 0.0049
```

With the normal prior distribution, this probability will never be zero, but may be acceptably small, so we can still use the conjugate normal gamma model for analysis.

We can use the same strategy to generate samples from the predictive distribution of a new measurement $Y_{n+1}$ given the observed data. In mathematical terms, the posterior predictive distribution is written as

$$Y_{n+1} \mid Y_1, \ldots, Y_n \sim \St(v_n, m_n, s^2_n (1 + 1/n_n))$$

In the code, we replace the prior hyper parameters with the posterior hyper parameters from last time.


```r
set.seed(1234)
phi = rgamma(10000, v_n/2, s2_n*v_n/2)
sigma = 1/sqrt(phi)
post_mu = rnorm(10000, mean=m_n, sd=sigma/(sqrt(n_n)))
pred_y =  rnorm(10000,post_mu, sigma)
quantile(pred_y, c(.025, .975))
```

```
##      2.5%     97.5% 
##  3.280216 89.830212
```

Figure \@ref(fig:hist-pred) shows the Monte Carlo approximation to the prior distribution of $\mu$, and the posterior distribution of $\mu$ which is shifted to the right. The prior and posterior predictive distributions are also depicted, showing how the data have updated the prior information.

\begin{figure}

{\centering \includegraphics{04-normalgamma-03-predictive_files/figure-latex/hist-pred-1} 

}

\caption{Posterior densities}(\#fig:hist-pred)
\end{figure}

Using the Monte-Carlo samples from the posterior predictive distribution, we can estimate the probability that a new TTHM sample will exceed the legal limit of 80 parts per billion, which is approximately 0.06.


```r
sum(pred_y > 80)/length(pred_y)  # P(Y > 80 | data)
```

```
## [1] 0.0619
```

By using Monte-Carlo methods, we can obtain prior and posterior predictive distributions of the data.

* Sampling from the prior predictive distribution can help with the selection of prior hyper parameters and verify that these choices reflect the prior information that is available.

* Visualizing prior predictive distributions based on Monte Carlo simulations can help explore implications of our prior assumptions such as the choice of the hyper parameters or even assume distributions.

* If samples are incompatible with known information, such as support on positive values, we may need to modify assumptions and look at other families of prior distributions.
