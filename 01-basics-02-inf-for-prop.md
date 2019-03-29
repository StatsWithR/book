## Inference for a Proportion

### Inference for a Proportion: Frequentist Approach

\BeginKnitrBlock{example}<div class="example"><span class="example" id="exm:RU-486"><strong>(\#exm:RU-486) </strong></span>RU-486 is claimed to be an effective "morning after" contraceptive pill, but is it really effective?

Data: A total of 40 women came to a health clinic asking for emergency contraception (usually to prevent pregnancy after unprotected sex). They were randomly assigned to RU-486 (treatment) or standard therapy (control), 20 in each group. In the treatment group, 4 out of 20 became pregnant. In the control group, the pregnancy rate is 16 out of 20.

Question: How strongly do these data indicate that the treatment is more effective than the control?</div>\EndKnitrBlock{example}

To simplify the framework, let's make it a one proportion problem and just consider the 20 total pregnancies because the two groups have the same sample size. If the treatment and control are equally effective, then the probability that a pregnancy comes from the treatment group ($p$) should be 0.5. If RU-486 is more effective, then the probability that a pregnancy comes from the treatment group ($p$) should be less than 0.5.

Therefore, we can form the hypotheses as below:

* $p =$ probability that a given pregnancy comes from the treatment group

* $H_0: p = 0.5$ (no difference, a pregnancy is equally likely to come from the treatment or control group)

* $H_A: p < 0.5$ (treatment is more effective, a pregnancy is less likely to come from the treatment group)
 
A p-value is needed to make an inference decision with the frequentist approach. The definition of p-value is the probability of observing something **at least** as extreme as the data, given that the null hypothesis ($H_0$) is true. "More extreme" means in the direction of the alternative hypothesis ($H_A$).

Since $H_0$ states that the probability of success (pregnancy) is 0.5, we can calculate the p-value from 20 independent Bernoulli trials where the probability of success is 0.5. The outcome of this experiment is 4 successes in 20 trials, so the goal is to obtain 4 or fewer successes in the 20 Bernoulli trials. 

This probability can be calculated exactly from a binomial distribution with $n=20$ trials and success probability $p=0.5$. Assume $k$ is the actual number of successes observed, the p-value is 

$$P(k \leq 4) = P(k = 0) + P(k = 1) + P(k = 2) + P(k = 3) + P(k = 4)$$.


```r
sum(dbinom(0:4, size = 20, p = 0.5))
```

```
## [1] 0.005908966
```

According to $\mathsf{R}$, the probability of getting 4 or fewer successes in 20 trials is 0.0059. Therefore, given that pregnancy is equally likely in the two groups, we get the chance of observing 4 or fewer preganancy in the treatment group is 0.0059. With such a small probability, we reject the null hypothesis and conclude that the data provide convincing evidence for the treatment being more effective than the control. 

### Inference for a Proportion: Bayesian Approach

This section uses the same example, but this time we make the inference for the proportion from a Bayesian approach. Recall that we still consider only the 20 total pregnancies, 4 of which come from the treatment group. The question we would like to answer is that how likely is for 4 pregnancies to occur in the treatment group. Also remember that if the treatment and control are equally effective, and the sample sizes for the two groups are the same, then the probability ($p$) that the pregnancy comes from the treatment group is 0.5. 

Within the Bayesian framework, we need to make some assumptions on the models which generated the data. First, $p$ is a probability, so it can take on any value between 0 and 1. However, let's simplify by using discrete cases -- assume $p$, the chance of a pregnancy comes from the treatment group, can take on nine values, from 10%, 20%, 30%, up to 90%. For example, $p = 20\%$ means that among 10 pregnancies, it is expected that 2 of them will occur in the treatment group. Note that we consider all nine models, compared with the frequentist paradigm that whe consider only one model. 

Table \@ref(tab:RU-486prior) specifies the prior probabilities that we want to assign to our assumption. There is no unique correct prior, but any prior probability should reflect our beliefs prior to the experiement. The prior probabilities should incorporate the information from all relevant research before we perform the current experiement.

\begin{table}[t]

\caption{(\#tab:RU-486prior)Prior, likelihood, and posterior probabilities for each of the 9 models}
\centering
\begin{tabular}{lrrrrrrrrr}
\toprule
Model (p) & 0.1000 & 0.2000 & 0.3000 & 0.4000 & 0.5000 & 6e-01 & 0.70 & 0.80 & 0.90\\
Prior P(model) & 0.0600 & 0.0600 & 0.0600 & 0.0600 & 0.5200 & 6e-02 & 0.06 & 0.06 & 0.06\\
Likelihood P(data|model) & 0.0898 & 0.2182 & 0.1304 & 0.0350 & 0.0046 & 3e-04 & 0.00 & 0.00 & 0.00\\
P(data|model) x P(model) & 0.0054 & 0.0131 & 0.0078 & 0.0021 & 0.0024 & 0e+00 & 0.00 & 0.00 & 0.00\\
Posterior P(model|data) & 0.1748 & 0.4248 & 0.2539 & 0.0681 & 0.0780 & 5e-04 & 0.00 & 0.00 & 0.00\\
\bottomrule
\end{tabular}
\end{table}

This prior incorporates two beliefs: the probability of $p = 0.5$ is highest, and the benefit of the treatment is symmetric. The second belief means that the treatment is equally likely to be better or worse than the standard treatment. Now it is natural to ask how I came up with this prior, and the specification will be discussed in detail later in the course.

Next, let's calculate the likelihood -- the probability of observed data for each model considered. In mathematical terms, we have

$$ P(\text{data}|\text{model}) = P(k = 4 | n = 20, p)$$

The likelihood can be computed as a binomial with 4 successes and 20 trials with $p$ is equal to the assumed value in each model. The values are listed in Table \@ref(tab:RU-486prior).

After setting up the prior and computing the likelihood, we are ready to calculate the posterior using the Bayes' rule, that is,

$$P(\text{model}|\text{data}) = \frac{P(\text{model})P(\text{data}|\text{model})}{P(\text{data})}$$

The posterior probability values are also listed in Table \@ref(tab:RU-486prior), and the highest probability occurs at $p=0.2$, which is 42.48%. Note that the priors and posteriors across all models both sum to 1.

In decision making, we choose the model with the highest posterior probability, which is $p=0.2$. In comparison, the highest prior probability is at $p=0.5$ with 52%, and the posterior probability of $p=0.5$ drops to 7.8%. This demonstrates how we update our beliefs based on observed data. Note that the calculation of posterior, likelihood, and prior is unrelated to the frequentist concept (data "at least as extreme as observed").


Here are the histograms of the prior, the likelihood, and the posterior probabilities:

![(\#fig:RU-486plot)Original: sample size $n=20$ and number of successes $k=4$](01-basics-02-inf-for-prop_files/figure-latex/RU-486plot-1.pdf) 

We started with the high prior at $p=0.5$, but the data likelihood peaks at $p=0.2$. And we updated our prior based on observed data to find the posterior. The Bayesian paradigm, unlike the frequentist approach, allows us to make direct probability statements about our models. For example, we can calculate the probability that RU-486, the treatment, is more effective than the control as the sum of the posteriors of the models where $p<0.5$. Adding up the relevant posterior probabilities in Table \@ref(tab:RU-486prior), we get the chance that the treatment is more effective than the control is 92.16%. 

### Effect of Sample Size on the Posterior

The RU-486 example is summarized in Figure \@ref(fig:RU-486plot), and let's look at what the posterior distribution would look like if we had more data.

![(\#fig:RU-486plotX2)More data: sample size $n=40$ and number of successes $k=8$](01-basics-02-inf-for-prop_files/figure-latex/RU-486plotX2-1.pdf) 

Suppose our sample size was 40 instead of 20, and the number of successes was 8 instead of 4. Note that the ratio between the sample size and the number of successes is still 20%. We will start with the same prior distribution. Then calculate the likelihood of the data which is also centered at 0.20, but is less variable than the original likelihood we had with the smaller sample size. And finally put these two together to obtain the posterior distribution. The posterior also has a peak at p is equal to 0.20, but the peak is taller, as shown in Figure \@ref(fig:RU-486plotX2). In other words, there is more mass on that model, and less on the others. 

![(\#fig:RU-486plotX10)More data: sample size $n=200$ and number of successes $k=40$](01-basics-02-inf-for-prop_files/figure-latex/RU-486plotX10-1.pdf) 

To illustrate the effect of the sample size even further, we are going to keep increasing our sample size, but still maintain the the 20% ratio between the sample size and the number of successes. So let's consider a sample with 200 observations and 40 successes. Once again, we are going to use the same prior and the likelihood is again centered at 20% and almost all of the probability mass in the posterior is at p is equal to 0.20. The other models do not have zero probability mass, but they're posterior probabilities are very close to zero. 

Figure \@ref(fig:RU-486plotX10) demonstrates that **as more data are collected, the likelihood ends up dominating the prior**. This is why, while a good prior helps, a bad prior can be overcome with a large sample. However, it's important to note that this will only work as long as we do not place a zero probability mass on any of the models in the prior. 
