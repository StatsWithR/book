---
output:
  pdf_document: default
  html_document: default
---
## Comparing Two Paired Means using Bayes Factors

We previously learned that we can use a paired t-test to compare means from two paired samples. In this section, we will show how Bayes factors can be expressed as a function of the t-statistic for comparing the means and provide posterior probabilities of the hypothesis that whether the means are equal or different.

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:zinc"><strong>(\#exm:zinc) </strong></span>Trace metals in drinking water affect the flavor, and unusually high concentrations can pose a health hazard. Ten pairs of data were taken measuring the zinc concentration in bottom and surface water at ten randomly sampled locations, as listed in Table \@ref(tab:zinc-table).

Water samples collected at the the same location, on the surface and the bottom, cannot be assumed to be independent of each other. However, it may be reasonable to assume that the differences in the concentration at the bottom and the surface in randomly sampled locations are independent of each other.</div>\EndKnitrBlock{example}

\begin{table}

\caption{(\#tab:zinc-table)Zinc in drinking water}
\centering
\begin{tabular}[t]{rrrr}
\toprule
location & bottom & surface & difference\\
\midrule
1 & 0.430 & 0.415 & 0.015\\
2 & 0.266 & 0.238 & 0.028\\
3 & 0.567 & 0.390 & 0.177\\
4 & 0.531 & 0.410 & 0.121\\
5 & 0.707 & 0.605 & 0.102\\
\addlinespace
6 & 0.716 & 0.609 & 0.107\\
7 & 0.651 & 0.632 & 0.019\\
8 & 0.589 & 0.523 & 0.066\\
9 & 0.469 & 0.411 & 0.058\\
10 & 0.723 & 0.612 & 0.111\\
\bottomrule
\end{tabular}
\end{table}

To start modeling, we will treat the ten differences as a random sample from a normal population where the parameter of interest is the difference between the average zinc concentration at the bottom and the average zinc concentration at the surface, or the main difference, $\mu$.

In mathematical terms, we have

* Random sample of $n= 10$ differences $Y_1, \ldots, Y_n$
* Normal population with mean $\mu \equiv  \mu_B - \mu_S$

In this case, we have no information about the variability in the data, and we will treat the variance, $\sigma^2$, as unknown.

The hypothesis of the main concentration at the surface and bottom are the same is equivalent to saying $\mu = 0$. The second hypothesis is that the difference between the mean bottom and surface concentrations, or equivalently that the mean difference $\mu \neq 0$.

In other words, we are going to compare the following hypotheses:

* $H_1: \mu_B = \mu_S  \Leftrightarrow \mu = 0$
* $H_2: \mu_B \neq \mu_S \Leftrightarrow \mu  \neq 0$

The Bayes factor is the ratio between the distributions  of the data
under each hypothesis, which does not depend on any unknown parameters.

$$\BF[H_1 : H_2] = \frac{p(\data \mid H_1)} {p(\data \mid H_2)}$$

To obtain the Bayes factor, we need to use integration over the prior distributions under each hypothesis to obtain those distributions of the data.

$$\BF[H_1 : H_2] = \iint p(\data \mid \mu, \sigma^2) p(\mu \mid \sigma^2) p(\sigma^2 \mid H_2)\, d \mu \, d\sigma^2$$

This requires specifying the following priors:

* $\mu \mid \sigma^2, H_2 \sim \No(0, \sigma^2/n_0)$
* $p(\sigma^2) \propto 1/\sigma^2$ for both $H_1$ and $H_2$

$\mu$ is exactly zero under the hypothesis $H_1$. For $\mu$ in $H_2$, we start with the same conjugate normal prior as we used in Section \@ref(sec:known-var) -- testing the normal mean with known variance. Since we assume that $\sigma^2$ is known, we model $\mu \mid \sigma^2$ instead of $\mu$ itself.

The $\sigma^2$ appears in both the numerator and denominator of the Bayes factor. For default or reference case, we use the Jeffreys prior (a.k.a. reference prior) on $\sigma^2$. As long as we have more than two observations, this (improper) prior will lead to a proper posterior.

After integration and rearranging, one can derive a simple expression for the Bayes factor:

$$\BF[H_1 : H_2] = \left(\frac{n + n_0}{n_0} \right)^{1/2} \left(
  \frac{ t^2  \frac{n_0}{n + n_0} + \nu }
  { t^2  + \nu} \right)^{\frac{\nu + 1}{2}}$$

This is a function of the t-statistic

$$t = \frac{|\bar{Y}|}{s/\sqrt{n}},$$

where $s$ is the sample standard deviation and the degrees of freedom $\nu = n-1$ (sample size minus one).

As we saw in the case of Bayes factors with known variance, we cannot use the improper prior on $\mu$ because when $n_0 \to 0$, then $\BF[H1:H_2] \to \infty$ favoring $H_1$ regardless of the magnitude of the t-statistic. Arbitrary, vague small choices for $n_0$ also lead to arbitrary large Bayes factors in favor of $H_1$. Another example of the Barlett's or Jeffreys-Lindley paradox.

Sir Herald Jeffrey discovered another paradox testing using the conjugant normal prior, known as the **information paradox**. His thought experiment assumed that our sample size $n$ and the prior sample size $n_0$. He then considered what would happen to the Bayes factor as the sample mean moved further and further away from the hypothesized mean, measured in terms standard errors with the t-statistic, i.e., $|t| \to \infty$. As the t-statistic or information about the mean moved further and further from zero, the Bayes factor goes to a constant depending on $n, n_0$ rather than providing overwhelming support for $H_2$.

The bounded Bayes factor is

$$\BF[H_1 : H_2] \to \left( \frac{n_0}{n_0 + n}  \right)^{\frac{n - 1}{2}}$$

Jeffrey wanted a prior with $\BF[H_1 : H_2] \to 0$ (or equivalently, $\BF[H_2 : H_1] \to \infty$), as the information from the t-statistic grows, indicating the sample mean is as far as from the hypothesized mean and should favor $H_2$.

To resolve the paradox when the information the t-statistic favors $H_2$ but the Bayes factor does not, Jeffreys showed that **no normal prior could resolve the paradox**.

But a **Cauchy prior** on $\mu$, would resolve it. In this way, $\BF[H_2 : H_1]$ goes to infinity as the sample mean becomes further away from the hypothesized mean. Recall that the Cauchy prior is written as $\Ca(0, r^2 \sigma^2)$. While Jeffreys used a default of $r = 1$, smaller values of $r$ can be used if smaller effects are expected.

The combination of the Jeffrey's prior on $\sigma^2$ and this Cauchy prior on $\mu$ under $H_2$ is sometimes referred to as the **Jeffrey-Zellener-Siow prior**.

However, there is no closed form expressions for the Bayes factor under the Cauchy distribution. To obtain the Bayes factor, we must use the
numerical integration or simulation methods.

We will use the \texttt{bayes$\_$inference} function from the \texttt{statsr} package to test whether the mean difference is zero in Example \@ref(exm:zinc) (zinc), using the JZS (Jeffreys-Zellener-Siow) prior.


```r
library(statsr)
bayes_inference(difference, data=zinc, statistic="mean", type="ht",
                prior="JZS", mu_0=0, method="theo", alt="twosided")
```

```
## Single numerical variable
## n = 10, y-bar = 0.0804, s = 0.0523
## (Using Zellner-Siow Cauchy prior:  mu ~ C(0, 1*sigma)
## (Using Jeffreys prior: p(sigma^2) = 1/sigma^2
## 
## Hypotheses:
## H1: mu = 0 versus H2: mu != 0
## Priors:
## P(H1) = 0.5 , P(H2) = 0.5
## Results:
## BF[H2:H1] = 50.7757
## P(H1|data) = 0.0193  P(H2|data) = 0.9807 
## 
## Posterior summaries for mu under H2:
## Single numerical variable
## n = 10, y-bar = 0.0804, s = 0.0523
## (Assuming Zellner-Siow Cauchy prior:  mu | sigma^2 ~ C(0, 1*sigma)
## (Assuming improper Jeffreys prior: p(sigma^2) = 1/sigma^2
## 
## Posterior Summaries
##             2.5%        25%        50%         75%       97.5%
## mu    0.03646550 0.06326657 0.07534282  0.08715550  0.11215657
## sigma 0.03666592 0.04743897 0.05533829  0.06557523  0.09533053
## n_0   0.16443438 1.89693942 4.74700781 10.09536724 32.45855610
## 
## 95% CI for mu: (0.0365, 0.1122)
```



\begin{center}\includegraphics{05-BFnormal-02-paired-means_files/figure-latex/bayes-inference-1} \end{center}

With equal prior probabilities on the two hypothesis, the Bayes factor is the posterior odds. From the output, we see this indicates that the hypothesis $H_2$, the mean difference is different from 0, is almost 51 times more likely than the hypothesis $H_1$ that the average concentration is the same at the surface and the bottom.

To sum up, we have used the **Cauchy prior** as a default prior testing hypothesis about a normal mean when variances are unknown. This does require numerical integration, but it is available in the \texttt{bayes$\_$inference} function from the \texttt{statsr} package. If you expect that the effect sizes will be small, smaller values of $r$ are recommended.

It is often important to quantify the magnitude of the difference in addition to testing. The Cauchy Prior provides a default prior for both testing and inference; it avoids problems that arise with choosing a value of $n_0$ (prior sample size) in both cases. In the next section, we will illustrate using the Cauchy prior for comparing two means from independent normal samples.
