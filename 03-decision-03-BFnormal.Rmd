## Hypothesis Testing with Normal Populations

In Section \@ref(sec:bayes-factors), we described how the Bayes factors can be used for hypothesis testing. Now we will use the Bayes factors to test normal means, i.e. compare two groups of normally-distributed populations. We divide this mission into four cases: known variance, unknown variance, paired data, and independent groups.

### Bayes Factors for Testing a Normal Mean: variance known

Now we show how to obtain base factors for testing hypothesis about a normal mean, where **the variance is known**. To start, let's consider a random sample of observations from a normal population with mean $\mu$ and pre-specified variance $\sigma^2$. We consider testing whether the population mean $\mu$ is equal to $m_0$ or not. 

Therefore, we can formulate the data and hypotheses as below:

**Data**
$$Y_1, \cdots, Y_n \iid \No(\mu, \sigma^2)$$

**Hypotheses**

* $H_1: \mu = m_0$
* $H_2: \mu \neq m_0$

We also need to specify priors for $\mu$ under both hypotheses. Under $H_1$, we assume that $\mu$ is exactly $m_0$, so this occurs with probability 1 under $H_1$. Now under $H_2$, $\mu$ is unspecified, so we describe our prior uncertainty with the conjugate normal distribution centered at $m_0$ and with a variance $\sigma^2/\mathbf{n_0}$. This is centered at the hypothesized value $m_0$, and it seems that the mean is equally likely to be larger or smaller than $m_0$, so a dividing factor $n_0$ is given to the variance. The hyper parameter $n_0$ controls the precision of the prior as before.

In mathematical terms, the priors are:

**Priors**

* $H_1: \mu = m_0  \text{  with probability 1}$
* $H_2: \mu \sim \No(m_0, \sigma^2/\mathbf{n_0})$

UNFINISHED BELOW

**Bayes Factor**

$$\begin{aligned}
\BF[H_1 : H_2] &= \frac{p(\data \mid \mu = m_0, \sigma^2 )}
 {\int p(\data \mid \mu, \sigma^2) p(\mu \mid m_0, \mathbf{n_0}, \sigma^2)\, d \mu} \\ 
\BF[H_1 : H_2] &=\left(\frac{n + \mathbf{n_0}}{\mathbf{n_0}} \right)^{1/2} \exp\left\{-\frac 1 2 \frac{n }{n + \mathbf{n_0}} Z^2 \right\} \\
 Z   &=  \frac{(\bar{Y} - m_0)}{\sigma/\sqrt{n}}
\end{aligned}$$

### Bayes Factors for Testing a Normal Mean: unknown variance

### Testing Normal Means: paired data

### Testing Normal Means: independent groups

### Inference after Testing
