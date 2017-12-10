## Comparing Independent  Means: hypothesis testing

In the previous section, we described Bayes factors for testing whether the mean difference of **paired** samples was zero. In this section, we will consider a slightly different problem -- we have two **independent** samples, and we would like to test the hypothesis that the means are different or equal.

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:birth-records"><strong>(\#exm:birth-records) </strong></span>We illustrate the testing of independent groups with data from a 2004 survey of birth records from North Carolina, which are available in the \texttt{statsr} package.

The variable of interest is \texttt{gained} -- the weight gain of mothers during pregnancy. We have two groups defined by the categorical variable, \texttt{mature}, with levels, younger mom and older mom.

**Question of interest**: Do the data provide convincing evidence of a difference between the average weight gain of older moms and the average weight gain of younger moms?</div>\EndKnitrBlock{example}

We will view the data as a random sample from two populations, older and younger moms. The two groups are modeled as:

\begin{equation}
\begin{split}
Y_{O,i} &\mathrel{\mathop{\sim}\limits^{\rm iid}} \textsf{N}(\mu + \alpha/2, \sigma^2) \\
Y_{Y,i} &\mathrel{\mathop{\sim}\limits^{\rm iid}} \textsf{N}(\mu - \alpha/2, \sigma^2)
\end{split}
(\#eq:half-alpha)
\end{equation}

The model for weight gain for older moms using the subscript $O$, and it assumes that the observations $Y$ are independent and identically distributed, with a mean $\mu+\alpha/2$ and variance $\sigma^2$.

For the younger women, the observations with the subscript $Y$ are independent and identically distributed with a mean $\mu-\alpha/2$ and variance $\sigma^2$.

Using this representation of the means in the two groups, the difference in means simplifies to $\alpha$ -- the parameter of interest.

$$(\mu + \alpha/2)  - (\mu - \alpha/2) =  \alpha$$

You may ask, "Why don't we set the average weight gain of older women to $\mu+\alpha$, and the average weight gain of younger women to $\mu$?" We need the parameter $\alpha$ to be present in both $Y_{O,i}$ (the group of older women) and $Y_{Y,i}$ (the group of younger women).

We have the following competing hypotheses:

* $H_1: \alpha = 0 \Leftrightarrow$ The means are not different.
* $H_2: \alpha \neq 0 \Leftrightarrow$ The means are different.

In this representation, $\mu$ represents the overall weight gain for all women. (Does the model in Equation \@ref(eq:half-alpha) make more sense now?) To test the hypothesis, we need to specify prior distributions for $\alpha$ under $H_2$ (c.f. $\alpha = 0$ under $H_1$) and priors for $\mu,\sigma^2$ under both hypotheses.

UNFINISHED BELOW

Recall that the Bayes factor is the ratio of the distribution of the data under the two hypotheses.

$$\begin{aligned}
 \BF[H_1 : H_2] &=  \frac{p(\data \mid H_1)} {p(\data \mid H_2)} \\
  &= \frac{\iint p(\data \mid \alpha = 0,\mu,  \sigma^2 )p(\mu, \sigma^2 \mid H_1) \, d\mu \,d\sigma^2}
 {\int \iint p(\data \mid \alpha, \mu, \sigma^2) p(\alpha \mid \sigma^2) p(\mu, \sigma^2 \mid H_2) \, d \mu \, d\sigma^2 \, d \alpha}
\end{aligned}$$