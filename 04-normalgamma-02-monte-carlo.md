## Monte Carlo Inference  {#sec:NG-MC}





In Section \@ref(sec:normal-gamma), we showed how to obtain the conditional posterior distribution  for the mean of a normal population given the variance and the marginal posterior distribution of the precision (inverse variance). The marginal distribution of the mean, which "averaged over uncertainty" about the unknown variance could be obtained via integration, leading to the Student t distribution that was used for inference about the population mean.  However, what if we are interested in the distribution of the standard deviation $\sigma$ itself, or other transformations of the parameters? There may not be a closed-form expression for the distributions or they may be difficult to obtain.

It turns out that **Monte Carlo sampling**, however, is an easy way to make an inference about parameters, when we cannot analytically calculate distributions of parameters, expectations, or probabilities. Monte Carlo methods are computational algorithms that rely on repeated random sampling from distributions for making inferences. The name refers to the famous Monte Carlo Casino in Monaco, home to games of chance such as roulette.

### Monte Carlo Sampling

Let's start with a case where we know the posterior distribution. As a quick recap, recall that the joint posterior distribution for the mean $\mu$ and the precision $\phi = 1/\sigma^2$ under the conjugate prior for the Gaussian distribution is:

* Conditional posterior distribution for the mean
$$\mu \mid \data, \sigma^2  \sim  \No(m_n, \sigma^2/n_n)$$
* Marginal posterior distribution for the precision $\phi$ or inverse variance:
$$1/\sigma^2 = \phi \mid \data   \sim   \Ga(v_n/2,s^2_n v_n/2)$$
* Marginal posterior distribution for the mean $$\mu \mid \data \sim \St(v_n, m_n, s^2_n/n_n)$$


For posterior inference about $\phi$, we can generate $S$ random samples from the Gamma posterior distribution:

$$\phi^{(1)},\phi^{(2)},\cdots,\phi^{(S)} \iid \Ga(v_n/2,s^2_n v_n/2)$$

Recall that the term **iid** stands for **i**ndependent and **i**dentically **d**istributed. In other words, the $S$ draws of $\phi$ are independent and identically distributed from the gamma distribution.

We can use the empirical distribution of the $S$ samples  to approximate the actual posterior distribution. The sample mean of the $S$ random draws of $\phi$ can be used to approximate the posterior mean of $\phi$.
Likewise, we can calculate probabilities, quantiles and other functions using the $S$ samples from the posterior distribution. For example, if we want to calculate the posterior expectation of some function of $\phi$, written as $g(\phi)$, we can approximate that by taking the average of the function, and evaluate it at the $S$ draws of $\phi$, written as $\frac{1}{S}\sum^S_{i=1}g(\phi^{(i)})$.

The approximation to the expectation of the function, $E[g(\phi \mid \data)]$ improves 

$$\frac{1}{S}\sum^S_{i=1}g(\phi^{(i)}) \rightarrow E[g(\phi \mid \data)]$$
as the number of draws $S$ in the Monte Carlo simulation increases.

### Tap Water Example (continued)

We will apply this to the tap water example from  \@ref(sec:normal-gamma).   First, reload the data and calculate the posterior hyper-parameters if needed.


```r
# Prior
m_0 = 35;  n_0 = 25;  s2_0 = 156.25; v_0 = n_0 - 1
# Data
data(tapwater); Y = tapwater$tthm
ybar = mean(Y); s2 = var(Y); n = length(Y)
# Posterior Hyper-paramters
n_n = n_0 + n
m_n = (n*ybar + n_0*m_0)/n_n
v_n = v_0 + n
s2_n = ((n-1)*s2 + v_0*s2_0 + n_0*n*(m_0 - ybar)^2/n_n)/v_n
```


Before generating our Monte Carlo samples, we will set a random seed using the `set.seed` function in `R`, which takes a small integer argument. 


```r
set.seed(42)
```

This allows the results to be replicated if you re-run the simulation at a later time.  

To generate $1,000$ draws from the gamma posterior distribution using the hyper-parameters above, we use the `rgamma` function `R`



```r
phi = rgamma(1000, shape = v_n/2, rate=s2_n*v_n/2)
```

The first argument to the `rgamma` function is the number of samples, the second is the shape parameter and, by default, the third argument is the rate parameter.

The following code will produce a histogram of the Monte Carlo samples of $\phi$ and overlay the actual Gamma posterior density evaluated at the draws using the `dgamma` function in `R`.



```r
df = data.frame(phi = sort(phi))
df = mutate(df, 
            density = dgamma(phi, 
                             shape = v_n/2,
                             rate=s2_n*v_n/2))

ggplot(data=df, aes(x=phi)) + 
        geom_histogram(aes(x=phi, y=..density..), bins = 50) +
        geom_density(aes(phi, ..density..), color="black") +
        geom_line(aes(x=phi, y=density), color="orange") +
        xlab(expression(phi)) + theme_tufte()
```

\begin{figure}

{\centering \includegraphics{04-normalgamma-02-monte-carlo_files/figure-latex/phi-plot-1} 

}

\caption{Monte Carlo approximation of the posterior distribution of the precision from the tap water example}(\#fig:phi-plot)
\end{figure}

Figure \@ref(fig:phi-plot) shows the histogram of the $1,000$ draws of $\phi$ generated from the Monte Carlo simulation, representing the empirical distribution approximation to the gamma posterior distribution. The orange line represents the actual gamma posterior density, while the black line represents a *smoothed* version of the histogram.

We can estimate the posterior mean or a 95\% equal tail area credible region using the Monte Carlo samples 


```r
mean(phi)
```

```
## [1] 0.002165663
```

```r
quantile(phi, c(0.025, 0.975))
```

```
##        2.5%       97.5% 
## 0.001394921 0.003056304
```

The mean of a gamma random variable is the shape/rate, so we can compare the Monte Carlo estimates to the theoretical values


```r
# mean  (v_n/2)/(v_n*s2_n/2)
1/s2_n
```

```
## [1] 0.002174492
```

```r
qgamma(c(0.025, 0.975), shape=v_n/2, rate=s2_n*v_n/2)
```

```
## [1] 0.001420450 0.003086519
```
where the `qgamma` function in `R` returns the desired quantiles provided as the first argument.
We can see that we can estimate the mean accurately to three significant digits, while the quantiles are accurate to two.  It increase our accuracy, we would need to increase $S$.

**Exercise**
Try increasing the number of simulations $S$ in the Monte Carlo simulation to $10,000$, and see how the approximation changes.

### Monte Carlo Inference for Functions of Parameters

Let's see how to use Monte Carlo simulations to approximate the distribution of $\sigma$. Since $\sigma = 1/\sqrt{\phi}$, we simply apply the transformation to the $1,000$ draws of $\phi$ to obtain a random sample of $\sigma$ from its posterior distribution. We can then estimate the posterior mean of $\sigma$ by calculating the sample mean of the 1,000 draws.


```r
sigma = 1/sqrt(phi)
mean(sigma) # posterior mean of sigma
```

```
## [1] 21.80516
```

Similarly, we can obtain a 95% credible interval for $\sigma$ by finding the sample quantiles of the distribution.


```r
quantile(sigma, c(0.025, 0.975))
```

```
##     2.5%    97.5% 
## 18.08847 26.77474
```
and finally approximate the posterior distribution using a smoothed density estimate


\begin{figure}

{\centering \includegraphics{04-normalgamma-02-monte-carlo_files/figure-latex/sigma-plot-1} 

}

\caption{Monte Carlo approximation of the posterior distribution of the standard deviation from the tap water example}(\#fig:sigma-plot)
\end{figure}

**Exercise**

Using the $10,000$ draws of $\phi$, create a histogram with a smoothed density overlay for the tap water example.

### Summary

To recap, we have introduced the powerful method of Monte Carlo simulation for posterior inference. Monte Carlo methods provide estimates of expectations, probabilities, and quantiles of distributions from the simulated values. Monte Carlo simulation also allows us to approximate distributions of functions of the parameters, or the transformations of the parameters.

Next we will discuss predictive distributions and show how Monte Carlo simulation may be used to help choose prior hyperparameters, using the prior predictive distribution of data.
