## Mixtures of Conjugate Priors  {#sec:NG-Cauchy}



In this section, we will describe priors that are constructed as a mixture of conjugate priors -- in particular, the Cauchy distribution. As these are no longer conjugate priors, nice analytic expressions for the posterior distribution are not available. However, we can use a Monte Carlo algorithm called Markov chain Monte Carlo (MCMC) for posterior inference.

In many situations, we may have reasonable prior information about the mean $\mu$, but we are less confident in how many observations our prior beliefs are equivalent to. We can address this uncertainty in the prior sample size, through an additional prior distribution on a $n_0$ via a hierarchical prior.

The hierarchical prior for the normal gamma distribution is written as
$$\begin{aligned}
\mu \mid \sigma^2, n_0 & \sim \No(m_0, \sigma^2/n_0) \\
n_0 \mid \sigma^2 &  \sim \Ga(1/2, r^2/2)
\end{aligned}$$

  If $r=1$, then this corresponds to a prior expected sample size of one because the expectation of $\Ga(1/2,1/2)$ is one.

The marginal prior distribution from $\mu$ can be attained via integration, and we get

$$\mu \mid \sigma^2  \sim  \Ca(m_0, \sigma^2 r^2)$$

  This is a **Cauchy distribution** centered at the prior mean $m_0$, with the scale parameter $\sigma^2 r^2$. The probability density function (pdf) is:

  $$p(\mu \mid \sigma) = \frac{1}{\pi \sigma r} \left( 1 +  \frac{(\mu - m_0)^2} {\sigma^2 r^2}  \right)^{-1}$$

  The Cauchy distribution does not have a mean or standard deviation, but the center (location) and the scale play a similar role to the mean and standard deviation of the normal distribution. The Cauchy distribution is a special case of a student $t$ distribution with one degree of freedom.

As Figure \@ref(fig:cauchy-plot) shows, the standard Cauchy distribution with $r=1$ and the standard normal distribution $\No(0,1)$ are centered at the same location. But the Cauchy distribution has heavier tails -- more probability on extreme values than the normal distribution with the same scale parameter $\sigma$. Cauchy priors were recommended by Sir Harold Jeffreys as a default objective prior for both estimation and testing.


\begin{figure}

{\centering \includegraphics{04-normalgamma-05-mixtures_files/figure-latex/cauchy-plot-1} 

}

\caption{Cauchy distribution}(\#fig:cauchy-plot)
\end{figure}

