## Three Conjugate Families

In this section, the three conjugate families are beta-binomial, normal-gamma, and normal-normal pairs. Each of them has its own applications in everyday life.

### Inference on a Binomial Proportion

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:RU-486more"><strong>(\#exm:RU-486more) </strong></span>Recall Example \@ref(exm:RU-486), a simplified version of a real clinical trial taken in Scotland. It concerned RU-486, a morning after pill that was being studied to determine whether it was effective at preventing unwanted pregnancies. It had 800 women, each of whom had intercourse no more than 72 hours before reporting to a family planning clinic to seek contraception. 

Half of these women were randomly assigned to the standard contraceptive, a large dose of estrogen and progesterone. And half of the women were assigned RU-486. Among the RU-486 group, there were no pregnancies. Among those receiving the standard therapy, four became pregnant. </div>\EndKnitrBlock{example}

Statistically, one can model these data as coming from a binomial distribution. Imagine a coin with two sides. One side is labeled standard therapy and the other is labeled RU-486. The coin was tossed four times, and each time it landed with the standard therapy side face up.

A frequentist would analyze the problem as below:

* The parameter $p$ is the probability of a preganancy comes from the standard treatment.

* $H_0: p \geq 0.5$ and $H_A: p < 0.5$

* The p-value is $0.5^4 = 0.0625 > 0.05$ 

Therefore, the frequentist fails to reject the null hypothesis, and will not conclude that RU-486 is superior to standard therapy.

Remark: The significance probability, or p-value, is the chance of observing
data that are as or more supportive of the alternative hypothesis than
the data that were collected, when the null hypothesis is true.

Now suppose a Bayesian performed the analysis. She may set her beliefs about the drug and decide that she has no prior knowledge about the efficacy of RU-486 at all. This would be reasonable if, for example, it were the first clinical trial of the drug. In that case, she would be using the uniform distribution on the interval from 0 to 1, which corresponds to the $\text{beta}(1,1)$ density. In mathematical terms,

$$p \sim \text{Unif}(0,1) = \text{beta}(1,1).$$

From conjugacy, we know that since there were four failures for RU-486 and no successes, that her posterior probability of an RU-486 child is 

$$p|x \sim \text{beta}(1+0,1+4) = \text{beta}(1,5).$$

This is a beta that has much more area near $p$ equal to 0. The mean of $\text{beta}(\alpha,\beta)$ is $\frac{\alpha}{\alpha+\beta}$. So this Bayesian now believes that the unknown $p$, the probability of an RU-468 child, is about 1 over 6. 

The standard deviation of a beta distribution with parameters in alpha and beta also has a closed form:

$$p \sim \text{beta}(\alpha,\beta) \Rightarrow \text{Standard deviation} = \sqrt{\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}}$$

Before she saw the data, the Bayesian's uncertainty expressed by her standard deviation was 0.71. After seeing the data, it was much reduced -- her posterior standard deviation is just 0.13. 

We promised not to do much calculus, so I hope you will trust me to tell you that this Bayesian now believes that her posterior probability that $p < 0.5$ is 0.96875. She thought there was a 50-50 chance that RU-486 is better. But now she thinks there's about a 97% chance that RU-486 is better. 

Suppose a fifth child were born, also to a mother who received standard chip therapy. Now the Bayesian's prior is beta(1, 5) and the additional data point further updates her to a new posterior beta of 1 and 6. **As data comes in, the Bayesian's previous posterior becomes her new prior, so learning is self-consistent.** 

This example has taught us several things: 

1. We saw how to build a statistical model for an applied problem. 

2. We could compare the frequentist and Bayesian approaches to inference and see large differences in the conclusions. 

3. We saw how the data changed the Bayesian's opinion with a new mean for p and less uncertainty. 

4. We learned that Bayesian's continually update as new data arrive. **Yesterday's posterior is today's prior.** 

### The Gamma-Poisson Conjugate Families

A second important case is the gamma-Poisson conjugate families. In this case the data come from a Poisson distribution, and the prior and posterior are both gamma distributions. 

The Poisson random variable can take any **non-negative integer value** all the way up to infinity. It is used in describing **count data**, where one counts the number of independent events that occur in a fixed amount of time, a fixed area, or a fixed volume. 

Moreover, the Poisson distribution has been used to describe the number of phone calls one receives in an hour. Or, the number of pediatric cancer cases in the city, for example, to see if pollution has elevated the cancer rate above that of in previous years or for similar cities. It is also used in medical screening for diseases, such as HIV, where one can count the number of T-cells in the tissue sample. 


The Poisson distribution has a single parameter $\lambda$, and it is denoted as $X \sim \text{Pois}(\lambda)$ with $\lambda>0$. The probability mass function is

$$P(X=k) = \frac{\lambda^k}{k!} \exp^{-\lambda} \text{ for } k=0,1,\cdots,$$

where $k! = k \times (k-1) \times \cdots \times 1$. This gives the probability of observing a random variable equal to $k$. 

Note that $\lambda$ is both the mean and the variance of the Poisson random variable. It is obvious that $\lambda$ must be greater than zero, because it represents the mean number of counts, and the variance should be greater than zero (except for constants, which have zero variance).


\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:Poisson"><strong>(\#exm:Poisson) </strong></span>Famously, von Bortkiewicz used the Poisson distribution to study the number of Prussian cavalrymen who were kicked to death by a horse each year. This is count data over the course of a year, and the events are probably independent, so the Poisson model makes sense.

He had data on 15 cavalry units for the 20 years between 1875 and 1894, inclusive. The total number of cavalrymen who died by horse kick was 200. 

One can imagine that a Prussian general might want to estimate $\lambda$. The average number per year, per unit. Perhaps in order to see whether some educational campaign about best practices for equine safety would make a difference.</div>\EndKnitrBlock{example}

Suppose the Prussian general is a Bayesian. Introspective elicitation leads him to think that $\lambda=0.75$ and standard deviation 1.

Modern computing was unavailable at that time yet, so the general will need to express his prior as a member of a family conjugate to the Poisson. It turns out that this family consists of the gamma distributions. Gamma distributions describe continuous non-negative random variables. As we know, the value of $\lambda$ in the Poisson can take any non-negative value so this fits. 

The gamma family is flexible, and Figure \@ref(fig:gamma) illustrates a wide range of gamma shapes. 

\begin{figure}

{\centering \includegraphics{02-inference-02-conjugate_files/figure-latex/gamma-1} 

}

\caption{Gamma family}(\#fig:gamma)
\end{figure}


The probability density function for the gamma is indexed by shape $k$ and scale $\theta$, denoted as $\text{Gamma}(k,\theta)$ with $k,\theta > 0$. The mathematical form of the distribution is 

$$f(x) = \dfrac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-x/\theta},$$
where

$$\Gamma(z) = \int^{\infty}_0 x^{z-1} e^{-x} dx.$$

$\Gamma(z)$, the gamma function, is simply a constant that ensures the area under curve between 0 and 1 sums to 1, just like in the beta probability distribution case of Equation \@ref(eq:beta). A special case is that $\Gamma(n) = (n-1)!$ when $n$ is a positive integer.

However, some books parameterize the gamma distribution in a slightly different way with shape $\alpha = k$ and rate (inverse scale) $\beta=1/\theta$:

$$f(x) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$$

For this example, we use the $k$-$\theta$ parameterization, but you should always check which parameterization is being used. For example, $\mathsf{R}$ uses the $\alpha$-$\beta$ parameterization by default.  
In the the later material we find that  using the rate parameterization is more convenient.

** ANY WAY TO MAKE THE MATERIAL MORE IN SYNC AS LABS LATER SECTIONS ALL USE THE RATE PARAMETERIZATION **

For our parameterization, the mean of $\text{Gamma}(k,\theta)$ is $k\theta$, and the variance is $k\theta^2$. We can get the general's prior as below:

$$\begin{aligned}
\text{Mean} &= k\theta = 0.75 \\
\text{Standard deviation} &= \theta\sqrt{k} = 1
\end{aligned}$$

Hence
$$k = \frac{9}{16} \text{ and } \theta = \frac{4}{3}$$

For the gamma Poisson conjugate family, suppose we observed data $x_1, x_2, \cdots, x_n$ that follow a Poisson distribution.Then similar to the previous section, we would recognize the kernel of the gamma when using the gamma-Poisson family. The posterior $\text{Gamma}(k^*, \theta^*)$ has parameters

$$k^* = k + \sum^n_{i=1} x_i \text{ and } \theta^* = \frac{\theta}{(n\theta+1)}.$$

For this dataset, $N = 15 \times 20 = 300$ observations, and the number of casualities is 200. Therefore, the general now thinks that the average number of Prussian cavalry officers who die at the hoofs of their horses follows a gamma distribution with the parameters below:

$$\begin{aligned}
k^* &= k + \sum^n_{i=1} x_i = \frac{9}{16} + 200 = 200.5625 \\
\theta^* = \frac{\theta}{(n\theta+1)} &= \frac{4/3}{300\times(4/3)} = 0.0033
\end{aligned}$$

How the general has changed his mind is described in Table \@ref(tab:before-after). After seeing the data, his uncertainty about lambda, expressed as a standard deviation, shrunk from 1 to 0.047.

\begin{table}

\caption{(\#tab:before-after)Before and after seeing the data}
\centering
\begin{tabular}[t]{lrr}
\toprule
  & lambda & Standard Deviation\\
\midrule
Before & 0.75 & 1.000\\
After & 0.67 & 0.047\\
\bottomrule
\end{tabular}
\end{table}

In summary, we learned about the Poisson and gamma distributions; we also knew that the gamma-Poisson families are conjugate. Moreover, we learned the updating fomula, and applied it to a classical dataset.

###  The Normal-Normal Conjugate Families {#sec:normal-normal}

There are other conjugate families, and one is the normal-normal pair. If your data come from a normal distribution with known standard deviation $\sigma$ but unknown mean $\mu$, and if your prior on the mean $\mu$, has a normal distribution with self-elicited mean $\nu$ and self-elicited standard deviation $\tau$, then your posterior density for the mean, after seeing a sample of size $n$ with sample mean $\bar{x}$, is also normal. In mathematical notation, we have

$$\begin{aligned}
x|\mu &\sim N(\mu,\sigma) \\
\mu &\sim N(\nu, \tau)
\end{aligned}$$

As a practical matter, one often does not know sigma, the standard deviation of the normal from which the data come. In that case, you could use a more advanced conjugate family that we will describe in \@ref(sec:normal-gamma). But there are cases in which it is reasonable to treat the $\sigma$ as known. 

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:chemist"><strong>(\#exm:chemist) </strong></span>An analytical chemist whose balance produces measurements that are normally distributed with mean equal to the true mass of the sample and standard deviation that has been estimated by the manufacturer balance and confirmed against calibration standards provided by the National Institute of Standards and Technology.

Note that this normal-normal assumption made by the anayltical chemist is technically wrong, but still reasonable.

1. The normal family puts some probability on all possible values between $(-\infty,+\infty)$. But the mass on the balance can **never** be negative. However, the normal prior on the unknown mass is usually so concentrated on positive values that the normal distribution is still a good approximation.

2. Even if the chemist has repeatedly calibrated her balance with standards from the National Institute of Standards and Technology, she still will not know its standard deviation precisely. However, if she has done it often and well, it is probably a sufficiently good approximation to assume that the standard deviation is known.
</div>\EndKnitrBlock{example}

For the normal-normal conjugate families, assume the prior on the unknown mean follows a normal distribution, i.e. $\mu \sim N(\nu, \tau)$. We also assume that the data $x_1,x_2,\cdots,x_n$ are independent and come from a normal with standard deviation $\sigma$.

Then the posterior distribution of $\mu$ is also normal, with mean as a weighted average of the prior mean and the sample mean. We have

$$\mu|x_1,x_2,\cdots,x_n \sim N(\nu^*, \tau^*),$$

where

$$\nu^* = \frac{\nu\sigma^2 + n\bar{x}\tau^2}{\sigma^2 + n\tau^2} \text{ and } \tau^* = \sqrt{\frac{\sigma^2\tau^2}{\sigma^2 + n\tau^2}}.$$

Let's continue from Example \@ref(exm:chemist), and suppose she wants to measure the mass of a sample of ammonium nitrate. 

Her balance has a known standard deviation of 0.2 milligrams. By looking at the sample, she thinks this mass is about 10 milligrams and based on her previous experience in estimating masses, her guess has the standard deviation of 2. So she decides that her prior for the mass of the sample is a normal distribution with mean, 10 milligrams, and standard deviation, 2 milligrams. 

Now she collects five measurements on the sample and finds that the average of those is 10.5. By conjugacy of the normal-normal family, our posterior belief about the mass of the sample has the normal distribution. 

The new mean of that posterior normal is found by plugging into the formula:

$$\begin{aligned}
\mu &\sim N(\nu=10, \tau=2) \\
\nu^*  &= \frac{\nu\sigma^2 + n\bar{x}\tau^2}{\sigma^2 + n\tau^2} = \frac{10\times(0.2)^2+5\times10.5\times2^2}{(0.2)^2+5\times2^2} = 10.499\\
\tau^* &= \sqrt{\frac{\sigma^2\tau^2}{\sigma^2 + n\tau^2}} = \sqrt{(0.2)^2\times2^2}{(0.2)^2+5\times2^2} = 0.089.
\end{aligned}$$

Before seeing the data, the Bayesian analytical chemist thinks the ammonium nitrate has mass 10 mg and uncertainty (standard deviation) 2 mg. After seeing the data, she thinks the mass is 10.499 mg and standard deviation 0.089 mg. Her posterior mean has shifted quite a bit and her uncertainty has dropped by a lot. That's exactly what an analytical chemist wants. 

This is the last of the three examples of conjugate families. There are many more, but they do not suffice for every situation one might have. 

We learned several things in this lecture. First, we learned the new pair of conjugate families and the relevant updating formula. Also, we worked a realistic example problem that can arise in practical situations. 
