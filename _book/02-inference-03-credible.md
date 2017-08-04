## Credible Intervals and Predictive Inference

Missing: JAGS plot for 2.3.1

### Non-Conjugate Priors

In many applications, a Bayesian may not be able to use a conjugate prior. Sometimes she may want to use a reference prior, which injects the minimum amount of personal belief into the analysis. But most often, a Bayesian will have a personal belief about the problem that cannot be expressed in terms of a convenient conjugate prior. 

For example, we shall reconsider the RU-486 case from earlier in which four children were born to standard therapy mothers. But no children were born to RU-486 mothers. This time, the Bayesian believes that the probability p of an RU-486 baby is uniformly distributed between 0 and one-half, but has a point mass of 0.5 at one-half. That is, she believes there's a 50% chance that there is no difference between standard therapy and RU-486. But if there is a difference, she thinks that RU-486 is better, but she is completely unsure about how much better it would be. 

In mathematical notation, the probability density function of $p$ is

$$f_p(x) = \left\{ \begin{array}{ccc}
1 & \text{for} & 0 \leq x < 0.5 \\
1 & \text{for} & x = 0.5 \\
0 & \text{for} & x < 0 \text{ or } x > 0.5
\end{array}\right.$$

We can check that the area under the density curve, plus the amount of the point mass, equals 1. 

The cumulative distribution function of $p$ is 

$$F_p(p \leq x) = \left\{ \begin{array}{ccc}
0 & \text{for} & x < 0 \\
x & \text{for} & 0 \leq x < 0.5  \\
1 & \text{for} & x \geq 0.5
\end{array}\right.$$

Why would this be a reasonable prior for an analyst to self-elicit? One reason is that in clinical trials, there's actually quite a lot of preceding research on the efficacy of the drug. This research might be based on animal studies or knowledge of the chemical activity of the molecule. So the Bayesian might feel sure that there is no possibility that RU-486 is worse than the standard treatment. And her interest is on whether the therapies are equivalent and if not, how much better RU-486 is than the standard therapy. 

As previously mentioned, there is no way to compute the posterior distribution for $p$ in a simple or even a complex mathematical form. And that is why Bayesian inference languished for so many decades until computational power enabled numerical solutions. But now we have such tools, and one of them is called **JAGS (Just Another Gibbs Sampler)**.

If we apply JAGS to the RU-486 data with this non-conjugate prior, we can find the posterior distribution. At a high level, this program is defining the binomial probability, that is the likelihood of seeing 0 RU-486 children, which is binomial. And then it defines the prior by using a few tricks to draw from either a uniform on the interval from 0 to one-half, or else draw from the point mass at one-half. Then it calls the JAGS model function, and draws 5,000 times from the posterior and creates a histogram of the results. 

NEED TO GET THE JAGS PLOT HERE

That histogram is lightly smooth to generate the posterior density you see. There is still a point mass of probability at 0.5, but now it has less weight than before. Also, note how the data have changed the posterior away from the prior. The analyst sees a lot of probability under the curve near 0.2, but responds to the fact that no children were born to RU-486 mothers. 

This section is mostly a look-ahead to future material. We have seen that a Bayesian might reasonably employ a non-conjugate prior in a practical application. But then she will need to employ some kind of numerical computation to approximate the posterior distribution. Additionally, we have used a computational tool, JAGS, to approximate the posterior for $p$, and identified its three important elements, the probability of the data given $p$, that is the likelihood, and the prior, and the call to the Gibbs sampler. 

### Credible Intervals

### Predictive Inference
