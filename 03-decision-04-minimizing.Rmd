## Minimizing Expected Loss for Hypothesis Testing

In Bayesian statistics, the inference about a parameter is made based on the posterior distribution, and let's include this in the hypothesis test setting.

Suppose we have two competing hypothesis, $H_1$ and $H_2$. Then we get

* $P(H_1 \text{ is true } | \text{ data})$ = posterior probability of $H_1$
* $P(H_2 \text{ is true } | \text{ data})$ = posterior probability of $H_2$

One straightforward way of choosing between $H_1$ and $H_2$ would be to **choose the one with the higher posterior probability**. In other words, the potential decision criterion is to

* Reject $H_1$ if $P(H_1 \text{ is true } | \text{ data}) < P(H_2 \text{ is true } | \text{ data})$.

However, since hypothesis testing is a decision problem, we should also consider a loss function. Let's revisit the HIV testing example in Section \@ref(sec:diagnostic-testing), and suppose we want to test the two competing hypotheses below:

  * $H_1$: Patient does not have HIV
* $H_2$: Patient has HIV

These are the only two possibilities, so they are mutually exclusive hypotheses that cover the entire decision space.

We can define the loss function as $L(d)$ -- the loss that occurs when decision $d$ is made. Then the Bayesian testing procedure minimizes the posterior expected loss.

The possible decisions (actions) are:

  * $d_1$: Choose $H_1$ - decide that the patient does not have HIV
* $d_2$: Choose $H_2$ - decide that the patient has HIV

For each decision $d$, we might be right, or we might be wrong. If the decision is right, the loss $L(d)$ associated with the decision $d$ is zero, i.e. no loss. If the decision is wrong, the loss $L(d)$ associated with the decision $d$ is some positive value $w$.

For $d=d_1$, we have

* **Right**: Decide patient does not have HIV, and indeed they do not. $\Rightarrow L(d_1) = 0$
  * **Wrong**: Decide patient does not have HIV, but they do. $\Rightarrow L(d_1) = w_1$

  For $d=d_2$, we also have

* **Right**: Decide patient has HIV, and indeed they do. $\Rightarrow L(d_2) = 0$
  * **Wrong**: Decide patient has HIV, but they don’t $\Rightarrow L(d2) = w_2$

  The consequences of making a wrong decision $d_1$ or $d_2$ are different.

Wrong $d_1$ is a **false negative**:

  * We decide that patient does not have HIV when in reality they do.
* Potential consequences: no treatment and premature death! (severe)

Wrong $d_2$ is a **false positive**:

  * We decide that the patient has HIV when in reality they do not.
* Potential consequences: distress and unnecessary further investigation. (not ideal but less severe than the consequences of a false negative decision)

Let's put these definitions in the context of the HIV testing example with ELISA.

**Hypotheses**

* $H_1$: Patient does not have HIV
* $H_2$: Patient has HIV

**Decision**

* $d_1$: Choose $H_1$ - decide that the patient does not have HIV
* $d_2$: Choose $H_2$ - decide that the patient has HIV

**Losses**

* $L(d_1) = \left\{ \begin{array}{cc}
0 & \text{if $d_1$ is right}\\
w_1=1000 & \text{if $d_1$ is wrong}
\end{array}\right.$

* $L(d_2) = \left\{ \begin{array}{cc}
0 & \text{if $d_2$ is right}\\
w_2=10 & \text{if $d_2$ is wrong}
\end{array}\right.$

The values of $w_1$ and $w_2$ are arbitrarily chosen. But the important thing is that $w_1$, the loss associated with a false negative determination, is much higher than $w_2$, the loss associated with a false positive determination.

**Posteriors**

The plus sign means that our patient had tested positive on the ELISA.

* $P(H_1|+) \approx 0.88$ - the posterior probability of the patient **not** having HIV given positive ELISA result
* $P(H_2|+) \approx 0.12$ - the posterior probability of the patient having HIV given positive ELISA result, as the complement value of $P(H_1|+)$

**Expected losses**

* $E[L(d_1)] = 0.88 \times 0 + 0.12 \times 1000 = 120$
* $E[L(d_2)] = 0.88 \times 10 + 0.12 \times 0 = 8.8$

Since the expected loss for $d_2$ is lower, we should make this decision -- the patient has HIV.

Note that our decision is highly influenced by the losses we assigned to $d_1$ and $d_2$.


If the losses were symmetric, say $w_1 = w_2 = 10$, then the expected loss for $d_1$ becomes

$$E[L(d_1)] = 0.88 \times 0 + 0.12 \times 10 = 1.2,$$

while the expected loss for $d_2$ would not change. Therefore, we would choose $d_1$ instead; that is, we would decide that the patient does not have HIV.

To recap, Bayesian methodologies allow for the integration of losses into the decision making framework easily. And in Bayesian testing, we minimize the posterior expected loss.

