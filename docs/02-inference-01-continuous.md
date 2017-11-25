## Continuous Variables and Eliciting Probability Distributions

### From the Discrete to the Continuous

This section leads the reader from the discrete random variable to continuous random variables. Let's start with the binomial random variable such as the number of heads in ten coin tosses, can only take a discrete number of values -- 0, 1, 2, up to 10.

When the probability of a coin landing heads is $p$, the chance of getting $k$ heads in $n$ tosses is

$$P(X = k) = \left( \begin{array}{c} n \\ k \end{array} \right) p^k (1-p)^{n-k}$$.

This formula is called the **probability mass function** (pmf) for the binomial.

The probability mass function can be visualized as a histogram in Figure \@ref(fig:histogram). The area under the histogram is one, and the area of each bar is the probability of seeing a binomial random variable, whose value is equal to the x-value at the center of the bars base. 

\begin{figure}

{\centering \includegraphics{02-inference-01-continuous_files/figure-latex/histogram-1} 

}

\caption{Histogram of binomial random variable}(\#fig:histogram)
\end{figure}


In contrast, the normal distribution, a.k.a. Gaussian distribution or the bell-shaped curve, can take any numerical value in $(-\infty,+\infty)$. A random variable generated from a normal distribution because it can take a continuum of values. 

In general, if the set of possible values a random variable can take are separated points, it is a discrete random variable. But if it can take any value in some (possibly infinite) interval, then it is a continuous random variable. 

When the random variable is **discrete**, it has a **probability mass function** or pmf. That pmf tells us the probability that the random variable takes each of the possible values. But when the random variable is continuous, it has probability zero of taking any single value. (Hence probability zero does not equal to impossible, an event of probabilty zero can still happen.)

We can only talk about the probability of a continuous random variable lined within some interval. For example, suppose that heights are approximately normally distributed. The probability of finding someone who is exactly 6 feet tall at 0.0000 inches tall for an infinite number of 0s after the decimal point is 0. But we can easily calculate the probability of finding someone who is between 5'11" inches tall and 6'1" inches tall. 

A **continuous** random variable has a **probability density function** or pdf, instead of probability mass functions. The probability of finding someone whose height lies between 5'11" (71 inches) and 6'1" (73 inches) is the area under the pdf curve for height between those two values, as shown in the blue area of Figure \@ref(fig:pdf-auc).^[Code reference: http://www.statmethods.net/advgraphs/probability.html]

\begin{figure}

{\centering \includegraphics{02-inference-01-continuous_files/figure-latex/pdf-auc-1} 

}

\caption{Area under curve for the probability density function}(\#fig:pdf-auc)
\end{figure}

For example, a normal distribution with mean $\mu$ and standard deviation $\sigma$ (i.e., variance $\sigma^2$) is defined as

$$f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp[-\frac{1}{2\sigma^2}(x-\mu)^2],$$

where $x$ is any value the random variable $X$ can take. This is denoted as $X \sim N(\mu,\sigma^2)$, where $\mu$ and $\sigma^2$ are the parameters of the normal distribution.

Recall that a probability mass function assigns the probability that a random variable takes a specific value for the discrete set of possible values. The sum of those probabilities over all possible values must equal one. 

Similarly, a probability density function is any $f(x)$ that is non-negative and has area one underneath its curve. The pdf can be regarded as the limit of histograms made from its sample data. As the sample size becomes infinitely large, the bin width of the histogram shrinks to zero. 

There are infinite number of pmf's and an infinite number of pdf's. Some distributions are so important that they have been given names:

* Continuous: normal, uniform, beta, gamma

* Discrete: binomial, Poisson

Here is a summary of the key ideas in this section:

1. Continuous random variables exist and they can take any value within some possibly infinite range. 

2. The probability that a continuous random variable takes a specific value is zero.

3. Probabilities from a continuous random variable are determined by the density function with this non-negative and the area beneath it is one. 

4. We can find the probability that a random variable lies between two values ($c$ and $d$) as the area under the density function that lies between them. 

### Elicitation

Next, we introduce the concept of prior elicitation in base and statistics. Often, one has a belief about the distribution of one's data. You may think that your data come from a binomial distribution and in that case you typically know the $n$, the number of trials but you usually do not know $p$, the probability of success. Or you may think that your data come from a normal distribution. But you do not know the mean $\mu$ or the standard deviation $\sigma$ of the normal. Beside to knowing the distribution of one's data, you may also have beliefs about the unknown $p$ in the binomial or the unknown mean $\mu$ in the normal. 

Bayesians express their belief in terms of personal probabilities. These personal probabilities encapsulate everything a Bayesian knows or believes about the problem. But these beliefs must obey the laws of probability, and be consistent with everything else the Bayesian knows. 

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:200percent"><strong>(\#exm:200percent) </strong></span>You cannot say that your probability of passing this course is 200%, no matter how confident you are. A probability value must be between zero and one. (If you still think you have a probability of 200% to pass the course, you are definitely not going to pass it.)</div>\EndKnitrBlock{example}

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:binomial-data"><strong>(\#exm:binomial-data) </strong></span>You may know nothing at all about the value of $p$ that generated some binomial data. In which case any value between zero and one is equally likely, you may want to make an inference on the proportion of people who would buy a new band of toothpaste. If you have industry experience, you may have a strong belief about the value of $p$, but if you are new to the industry you would do nothing about $p$. In any value between zero and one seems equally like a deal. This major personal probability is the uniform distribution whose probably density function is flat, denoted as $\text{Unif}(0,1)$.</div>\EndKnitrBlock{example}

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:coin-toss"><strong>(\#exm:coin-toss) </strong></span>If you were tossing a coin, most people believed that the probability of heads is pretty close to half. They know that some coin are loaded and they know that some coins may have two heads or two tails. And they probably also know that coins are not perfectly balanced. Nonetheless, before they start to collect data by tossing the coin and counting the number of heads their belief is that values of $p$ near 0.5 are very likely, where's values of $p$ near 0 or 1 are very unlikely. </div>\EndKnitrBlock{example}

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:marriage"><strong>(\#exm:marriage) </strong></span>In real life, here are two ways to elicit a probability that you cousin will get married. A frequentist might go to the U.S. Census records and determine what proportion of people get married (or, better, what proportion of people of your cousin's ethnicity, education level, religion, and age cohort are married). In contrast, a Bayesian might think "My cousin is brilliant, attractive, and fun. The probability that my cousin gets married is really high -- probably around 0.97."</div>\EndKnitrBlock{example}

So a base angle sits to express their belief about the value of $p$ through a probability distribution, and a very flexible family of distributions for this purpose is the **beta family**. A member of the beta family is specified by two parameters, $\alpha$ and $\beta$; we denote this as $p \sim \text{beta}(\alpha, \beta)$. The probability density function is 

\begin{equation}
f(p) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} p^{\alpha-1} (1-p)^{\beta-1},
(\#eq:beta)
\end{equation}

where $0 \leq p \leq 1, \alpha>0, \beta>0$, and $\Gamma$ is a factorial:

$$\Gamma(n) = (n-1)! = (n-1) \times (n-2) \times \cdots \times 1$$

When $\alpha=\beta=1$, the beta distribution becomes a uniform distribution, i.e. the probabilty density function is a flat line. In other words, the uniform distribution is a special case of the beta family.

The expected value of $p$ is $\frac{\alpha}{\alpha+\beta}$, so $\alpha$ can be regarded as the prior number of successes, and $\beta$ the prior number of failures. When $\alpha=\beta$, then one gets a symmetrical pdf around 0.5. For large but equal values of $\alpha$ and $\beta$, the area under the beta probability density near 0.5 is very large. Figure \@ref(fig:beta) compares the beta distribution with different parameter values.

\begin{figure}

{\centering \includegraphics{02-inference-01-continuous_files/figure-latex/beta-1} 

}

\caption{Beta family}(\#fig:beta)
\end{figure}

These kinds of priors are probably appropriate if you want to infer the probability of getting heads in a coin toss. The beta family also includes skewed densities, which is appropriate if you think that $p$ the probability of success in ths binomial trial is close to zero or one. 

Bayes' rule is a machine to turn one's prior beliefs into posterior beliefs. With binomial data you start with whatever beliefs you may have about $p$, then you observe data in the form of the number of head, say 20 tosses of a coin with 15 heads. 

Next, Bayes' rule tells you how the data changes your opinion about $p$. The same principle applies to all other inferences. You start with your prior probability distribution over some parameter, then you use data to update that distribution to become the posterior distribution that expresses your new belief. 

These rules ensure that the change in distributions from prior to posterior is the uniquely rational solution. So, as long as you begin with the prior distribution that reflects your true opinion, you can hardly go wrong. 

However, expressing that prior can be difficult. There are proofs and methods whereby a rational and coherent thinker can self-illicit their true prior distribution, but these are impractical and people are rarely rational and coherent. 

The good news is that with the few simple conditions no matter what part distribution you choose. If enough data are observed, you will converge to an accurate posterior distribution. So, two bayesians, say the reference Thomas Bayes and the agnostic Ajay Good can start with different priors but, observe the same data. As the amount of data increases, they will converge to the same posterior distribution. 

Here is a summary of the key ideas in this section:

1. Bayesians express their uncertainty through probability distributions. 

2. One can think about the situation and self-elicit a probability distribution that approximately reflects his/her personal probability. 

3. One's personal probability should change according Bayes' rule, as new data are observed. 

4. The beta family of distribution can describe a wide range of prior beliefs. 


### Conjugacy

Next, let's introduce the concept of conjugacy in Bayesian statistics. 

Suppose we have the prior beliefs about the data as below:

* Binomial distribution $\text{Bin}(n,p)$ with $n$ known and $p$ unknown

* Prior belief about $p$ is $\text{beta}(\alpha,\beta)$

Then we observe $x$ success in $n$ trials, and it turns out the Bayes' rule implies that our new belief about the probability density of $p$ is also the beta distribution, but with different parameters. In mathematical terms,

\begin{equation}
p|x \sim \text{beta}(\alpha+x, \beta+n-x).
(\#eq:beta-binomial)
\end{equation}

This is an example of conjugacy. Conjugacy occurs when the **posterior distribution** is in the **same family** of probability density functions as the prior belief, but with **new parameter values**, which have been updated to reflect what we have learned from the data. 

Why are the beta binomial families conjugate? Here is a mathematical explanation.

Recall the discrete form of the Bayes' rule:

$$P(A_i|B) = \frac{P(B|A_i)P(A_i)}{\sum^n_{j=1}P(B|A_j)P(A_j)}$$

However, this formula does not apply to continuous random variables, such as the $p$ which follows a beta distribution, because the denominator sums over all possible values (must be finitely many) of the random variable. 

But the good news is that the $p$ has a finite range -- it can tak any value **only** between 0 and 1. Hence we can perform integration, which is a generalization of the summation. The Bayes' rule can also be written in continuous form as:

$$\pi^*(p|x) = \frac{P(x|p)\pi(p)}{\int^1_0 P(x|p)\pi(p) dp}.$$

This is analogus to the discrete form, since the integral in the denominator will also be equal to some constant, just like a summation. This constant ensures that the total area under the curve, i.e. the posterior density function, equals 1.

Note that in the numerator, the first term, $P(x|p)$, is the data likelihood -- the probability of observing the data given a specific value of $p$. The second term, $\pi(p)$, is the probability density function that reflects the prior belief about $p$.

In the beta-binomial case, we have $P(x|p)=\text{Bin}(n,p)$ and $\pi(p)=\text{beta}(\alpha,\beta)$.

Plugging in these distributions, we get

$$\begin{aligned}
\pi^*(p|x) &= \frac{1}{\text{some number}} \times P(x|p)\pi(p) \\
&= \frac{1}{\text{some number}} [\left( \begin{array}{c} n \\ x \end{array} \right) p^x (1-p)^{n-x}] [\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} p^{\alpha-1} (1-p)^{\beta-1}] \\
&= \frac{\Gamma(\alpha+\beta+n)}{\Gamma(\alpha+x)\Gamma(\beta+n-x)} \times p^{\alpha+x-1} (1-p)^{\beta+n-x-1}
\end{aligned}$$

Let $\alpha^* = \alpha + x$ and $\beta^* = \beta+n-x$, and we get

$$\pi^*(p|x) = \text{beta}(\alpha^*,\beta^*) = \text{beta}(\alpha+x, \beta+n-x),$$

same as the posterior formula in Equation \@ref(eq:beta-binomial).

We can recognize the posterior distribution from the numerator $p^{\alpha+x-1}$ and $(1-p)^{\beta+n-x-1}$. Everything else are just constants, and they must take the unique value, which is needed to ensure that the area under the curve between 0 and 1 equals 1. So they have to take the values of the beta, which has parameters $\alpha+x$ and $\beta+n-x$.

This is a cute trick. We can find the answer without doing the integral simply by looking at form of the numerator. 

Without conjugacy, one has to do the integral. Often, the integral is impossible to evaluate. That obstacle is the primary reason that most statistical theory in the 20th century was not Bayesian. The situation didn't change until modern computing allowed researchers to compute integrals numerically. 

In summary, some pairs of distributions are conjugate. If your prior is in one and your data comes from the other, then your posterior is in the same family as the prior, but with new parameters. We explored this in the context of the beta-binomial conjugate families. And we saw that conjugacy meant that we could apply the continuous version of Bayes' rule without having to do any integration. 
