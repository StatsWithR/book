## Markov Chain Monte Carlo (MCMC)  {#sec:NG-MCMC}




The Cauchy prior described in Section \@ref(sec:NG-Cauchy) is not a conjugate prior, and therefore, the posterior distribution from $(\mu \mid \sigma^2)$, is not a Cauchy or any well-known distribution. Fortunately, the conditional distribution of $(\mu, \sigma^2 \mid n_0, \data)$, is normal-gamma and easy to simulate from, as we learned in the previous sections. The conditional distribution of $(n_0 \mid \mu, \sigma^2, \data$) is a gamma distribution, also easy to simulate from the given $\mu, \sigma^2$.

It turns out that if we alternate generating Monte Carlo samples from these conditional distributions, the sequence of samples converges to samples from the joint distribution of $(\mu, \sigma^2, n_0)$, as the number of simulated values increases. The Monte Carlo algorithm we have just described is a special case of Markov chain Monte Carlo (MCMC), known as the Gibbs sampler.

Let's look at the pseudo code for the algorithm.




```r
# initialize MCMC
sigma2[1] = 1; n_0[1]=1; mu[1]=m_0

#draw from full conditional distributions
for (i in 2:S) {
  mu[i]     = p_mu(sigma2[i-1], n_0[i-1],  m_0, r, data)
  sigma2[i] = p_sigma2(mu[i], n_0[i-1],    m_0, r, data)
  n_0[i]    = p_n_0(mu[i], sigma2[i],      m_0, r, data)
}
```

We start with the initial values of each of the parameters for $i=1$. In theory, these can be completely arbitrary, as long as they are allowed values for the parameters.

For each iteration $i$, the algorithm will cycle through generating each parameter, given the **current** value of the other parameters. The functions \texttt{p$\_$mu}, \texttt{p$\_$sigma2}, and \texttt{p$\_$n$\_$0} return a simulated value from the respective distribution conditional on the inputs.

Whenever we update a parameter, we use the **new value** in the subsequent steps as the $n$ draws for $\sigma, n_0$. We will repeat this until we reach iteration $S$, leading to a dependent sequence of s draws from the joint posterior distribution.

Incorporating the tap water example in Section \@ref(sec:normal-gamma), we will use MCMC to generate samples under the Cauchy prior. We set 35 as the location parameter and $r=1$. To complete our prior specification, we use the Jeffrey's reference prior on $\sigma^2$. This combination is referred to as the Jeffrey's Zellner-Siow Cauchy prior or "JZS" in the BayesFactor branch of the R \texttt{statsr} package.





```r
bayes_inference(y=tthm, data=tapwater, statistic="mean",
                mu_0 = 35, rscale=1, prior="JZS",
                type="ci", method="sim")
```

```
## Single numerical variable
## n = 28, y-bar = 55.5239, s = 23.254
## (Assuming Zellner-Siow Cauchy prior:  mu | sigma^2 ~ C(35, 1*sigma)
## (Assuming improper Jeffreys prior: p(sigma^2) = 1/sigma^2
## 
## Posterior Summaries
##             2.5%       25%      50%      75%    97.5%
## mu    45.5713714 51.820910 54.87345 57.87171 64.20477
## sigma 18.4996738 21.810376 23.84572 26.30359 32.11330
## n_0    0.2512834  2.512059  6.13636 12.66747 36.37425
## 
## 95% CI for mu: (45.5714, 64.2048)
```



\begin{center}\includegraphics{04-normalgamma-06-MCMC_files/figure-latex/tapwater-inference-1} \end{center}

Using the \texttt{bayes$\_$inference} function from the \texttt{statsr} package, we can obtain summary statistics and a plot from the MCMC output -- not only $\mu$, but also inference about $\sigma^2$ and the prior sample size.

The posterior mean under the JZS model is much closer to the sample mean than what the normal gamma prior used previously. Under the informative normal gamma prior, the sample made a 55.5, about eight standard deviations above the mean -- a surprising value under the normal prior. Under the Cauchy prior, the informative prior location has much less influence.

This is **the robustness property of the Cauchy prior**, leading the posterior to put more weight on the sample mean than the prior mean, especially when the prior location is not close to the sample mean. We can see that the central 50% interval for $n_0$ is well below the value 25 used in the normal prior, which placed almost equal weight on the prior in sample mean.

Using the MCMC draws of $\mu, \sigma$, we can obtain Monte Carlo samples from the predictive distribution of $y$, by plugging $\mu$ and $\sigma$ into the corresponding functions. Figure \@ref(fig:hist-ref-pred) compares the posterior densities estimated from the simulative values of $\mu$ and the predicted draws of TTHM under the Jeffrey Zellner-Siow prior, and the informative normal prior from $\mu$ with $n_0 = 25$ and the reference prior on $\sigma^2$.

\begin{figure}

{\centering \includegraphics{04-normalgamma-06-MCMC_files/figure-latex/hist-ref-pred-1} 

}

\caption{Comparison of posterior densities}(\#fig:hist-ref-pred)
\end{figure}

To recap, we have shown how to create more flexible prior distributions, such as the Cauchy distribution using mixtures of conjugate priors. As the posterior distributions are not available in closed form, we demonstrated how MCMC can be used for inference using the hierarchical prior distribution. Starting in the late 1980's, MCMC algorithms have led to an exponential rise in the use of Bayes in methods, because complex models built through hierarchical distributions suddenly were tractable. The Cauchy prior is well-known for being robust prior mis-specifications. For example, having a prior mean that is far from the observed mean. This provides an alternative to the reference prior as a default or objective distribution that is proper.

In the next sections, we will return to Bayes factors and hypothesis testing where the Cauchy prior plays an important role.
