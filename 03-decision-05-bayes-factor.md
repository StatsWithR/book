## Posterior Probabilities of Hypotheses and Bayes Factors {#sec:bayes-factors}

In this section, we will continue with the HIV testing example to introduce the concept of Bayes factors. Earlier, we introduced the concept of priors and posteriors. The **prior odds** is defined as **the ratio of the prior probabilities of hypotheses**.

Therefore, if there are two competing hypotheses being considered, then the prior odds of $H_1$ to $H_2$ can be defined as $O[H_1:H_2]$, which is equal to $P(H_1)$ over probability of $P(H_2)$. In mathematical terms,

$$O[H_1:H_2] = \frac{P(H_1)}{P(H_2)}$$

  Similarly, the **posterior odds** is **the ratio of the two posterior probabilities of hypotheses**, written as

$$PO[H_1:H_2] = \frac{P(H_1|\text{data})}{P(H_2|\text{data})}$$

  Using Bayes' rule, we can rewrite the posterior probabilities as below:

$$\begin{aligned}
PO[H_1:H_2] &= \frac{P(H_1|\text{data})}{P(H_2|\text{data})} \\
&= \frac{(P(\text{data}|H_1) \times P(H_1)) / P(\text{data}))}{(P(\text{data}|H_2) \times P(H_2)) / P(\text{data}))} \\
&= \frac{(P(\text{data}|H_1) \times P(H_1))}{(P(\text{data}|H_2) \times P(H_2))} \\
&= \boxed{\frac{P(\text{data}|H_1)}{P(\text{data}|H_2)}} \times \boxed{\frac{P(H_1)}{P(H_2)}} \\
&= \textbf{Bayes factor} \times \textbf{prior odds}
\end{aligned}$$

In mathematical notation, we have

$$PO[H_1:H_2] = BF[H_1:H_2] \times O[H_1:H_2]$$

In other words, the posterior odds is the product of the Bayes factor and the prior odds for these two hypotheses.

The Bayes factor quantifies the evidence of data arising from $H_1$ versus $H_2$.

In a discrete case, the Bayes factor is simply the ratio of the likelihoods of the observed data under the two hypotheses, written as

$$BF[H_1:H_2] = \frac{P(\text{data}|H_1)}{P(\text{data}|H_2)}.$$

On the other hand, in a continuous case, the Bayes factor is the ratio of the marginal likelihoods, written as

$$BF[H_1:H_2] = \frac{\int P(\text{data}|\theta,H_1)d\theta}{\int P(\text{data}|\theta,H_2)d\theta}.$$

Note that $\theta$ is the set formed by all possible values of the model parameters.

In this section, we will stick with the simpler discrete case. And in upcoming sections, we will revisit calculating Bayes factors for more complicated models.

Let's return to the HIV testing example from earlier, where our patient had tested positive in the ELISA.

**Hypotheses**

  * $H_1$: Patient does not have HIV
* $H_2$: Patient has HIV

**Priors**

  The prior probabilities we place on these hypothesis came from the prevalence of HIV at the time in the general population. We were told that the prevalence of HIV in the population was 1.48 out of 1000, hence the prior probability assigned to $H_2$ is 0.00148. And the prior assigned to $H_1$ is simply the complement of this.

* $P(H_1) = 0.99852$ and $P(H_2) = 0.00148$

  The prior odds are

* $O[H_1:H_2] = \dfrac{P(H_1)}{P(H_2)} = \dfrac{0.99852}{0.00148} = 674.6757$

  **Posteriors**

  Given a positive ELISA result, the posterior probabilities of these hypotheses can also be calculated, and these are approximately 0.88 and 0.12. We will hold on to more decimal places in our calculations to avoid rounding errors later.

* $P(H_1|+) = 0.8788551$ and $P(H_2|+) = 0.1211449$

  The posterior odds are

* $PO[H_1:H_2] = \dfrac{P(H_1|+)}{P(H_2|+)} = \dfrac{0.8788551}{0.1211449} = 7.254578$

  **Bayes Factor**

  Finally, we can calculate the Bayes factor as the ratio of the posterior odds to prior odds, which comes out to approximately 0.0108. Note that in this simple discrete case the Bayes factor, it simplifies to the ratio of the likelihoods of the observed data under the two hypotheses.

$$\begin{aligned}
BF[H_1:H_2] &= \frac{PO[H_1:H_2]}{O[H_1:H_2]} = \frac{7.25457}{674.6757} \approx 0.0108 \\
&= \frac{P(+|H_1)}{P(+|H_2)} = \frac{0.01}{0.93} \approx 0.0108
\end{aligned}$$

  Alternatively, remember that the true positive rate of the test was 0.93 and the false positive rate was 0.01. Using these two values, the Bayes factor also comes out to approximately 0.0108.

So now that we calculated the Bayes factor, the next natural question is, what does this number mean? A commonly used scale for interpreting Bayes factors is proposed by @jeffreys1961theory, as in Table \@ref(tab:jeffreys1961). If the Bayes factor is between 1 and 3, the evidence against $H_2$ is not worth a bare mention. If it is 3 to 20, the evidence is positive. If it is 20 to 150, the evidence is strong. If it is greater than 150, the evidence is very strong.

\begin{table}

\caption{(\#tab:jeffreys1961)Interpreting the Bayes factor}
\centering
\begin{tabular}[t]{cc}
\toprule
BF[H\_1:H\_2] & Evidence against H\_2\\
\midrule
1 to 3 & Not worth a bare mention\\
3 to 20 & Positive\\
20 to 150 & Strong\\
> 150 & Very strong\\
\bottomrule
\end{tabular}
\end{table}

It might have caught your attention that the Bayes factor we calculated does not even appear on the scale. To obtain a Bayes factor value on the scale, we will need to change the order of our hypotheses and calculate $BF[H_2:H_1]$, i.e. the Bayes factor for $H_2$ to $H_1$. Then we look for evidence against $H_1$ instead.

We can calculate $BF[H_2:H_1]$ as a reciprocal of $BF[H_1:H_2]$ as below:

  $$BF[H_2:H_1] = \frac{1}{BF[H_1:H_2]} = \frac{1}{0.0108} = 92.59259$$

  For our data, this comes out to approximately 93. Hence the evidence against $H_1$ (the patient does not have HIV) is strong. Therefore, even though the posterior  for having HIV given a positive result, i.e. $P(H_2|+)$, was low, we would still decide that the patient has HIV, according to the scale based on a positive ELISA result.

An intuitive way of thinking about this is to consider not only the posteriors, but also the priors assigned to these hypotheses. Bayes factor is the ratio of the posterior odds to prior odds. While 12% is a low posterior probability for having HIV given a positive ELISA result, this value is still much higher than the overall prevalence of HIV in the population (in other words, the prior probability for that hypothesis).

Another commonly used scale for interpreting Bayes factors is proposed by @kass1995bayes, and it deals with the natural logarithm of the calculated Bayes factor. The values can be interpreted in Table \@ref(tab:kass1995).

\begin{table}

\caption{(\#tab:kass1995)Interpreting the Bayes factor}
\centering
\begin{tabular}[t]{cc}
\toprule
2*log(BF[H\_2:H\_1]) & Evidence against H\_1\\
\midrule
0 to 2 & Not worth a bare mention\\
2 to 6 & Positive\\
6 to 10 & Strong\\
> 10 & Very strong\\
\bottomrule
\end{tabular}
\end{table}

Reporting of the log scale can be helpful for numerical accuracy reasons when the likelihoods are very small. Taking two times the natural logarithm of the Bayes factor we calculated earlier, we would end up with the same decision that the evidence against $H_1$ is strong.

$$2 \times \log(92.59259) = 9.056418$$

  To recap, we defined prior odds, posterior odds, and the Bayes factor. We learned about scales by which we can interpret these values for model selection. We also re-emphasize that in Bayesian testing, the order in which we evaluate the models of hypotheses does **not** matter. The Bayes factor of $H_2$ versus $H_1$, $BF[H_2:H_1]$, is simply the reciprocal of the Bayes factor for $H_1$ versus $H_2$, that is, $BF[H_1:H_2]$.
