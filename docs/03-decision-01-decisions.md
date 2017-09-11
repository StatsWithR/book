## Losses and Decision Making

To a Bayesian, the posterior distribution is the basis of any inference, since it integrates both his/her prior opinions and knowledge and the new information provided by the data. It also contains everything she believes about the distribution of the unknown parameter of interest. 

However, the posterior distribution on its own is not always sufficient. Sometimes the inference we want to express is a **credible interval**, because it indicates a range of likely values for the parameter. That would be helpful if you wanted to say that you are **95% certain** the probability of an RU-486 pregnancy lies between some number $L$ and some number $U$. And on other occasions, one needs to make a single number guess about the value of the parameter. For example, you might want to declare the average payoff for an insurance claim or tell a patient how much longer he/she has to live. 

Therefore, the Bayesian perspective leads directly to **decision theory**. And in decision theory, one seeks to minimize one's expected loss. 

### Loss Functions

Quantifying the loss can be tricky, and Table \@ref(tab:loss-functions) summarizes three different examples with three different loss functions.

If you're declaring the average payoff for an insurance claim, and if you are **linear** in how you value money, that is, twice as much money is exactly twice as good, then one can prove that the optimal one-number estimate is the **median** of the posterior distribution. But in different situations, other measures of loss may apply. 

If you are advising a patient on his/her life expectancy, it is easy to imagine that large errors are far more problematic than small ones. And perhaps the loss increases as the **square** of how far off your single number estimate is from the truth. For example, if she's told that her average life expectancy is two years, and it is actually ten, then her estate planning will be catastrophically bad, and she will die in poverty. In the case when the loss is proportional to the **quadratic** error, one can show that the optimal one-number estimate is the **mean** of the posterior distribution. 

Finally, in some cases, the penalty is 0 if you are exactly correct, but constant if you're at all wrong. This is the case with the old saying that close only counts with horseshoes and hand grenades; i.e., coming close but not succeeding is not good enough. And it would apply if you want a prize for correctly guessing the number of jelly beans in a jar. Here, of course, instead of minimizing expected losses, we want to **maximize the expected gain**. If a Bayesian is in such a situation, then his/her best one-number estimate is the **mode** of his/her posterior distribution, which is the most likely value. 

There is a large literature on decision theory, and it is directly linked to risk analysis, which arises in many fields. Although it is possible for frequentists to employ a certain kind of decision theory, it is much more natural for Bayesians. 



Table: (\#tab:loss-functions)Loss Functions

   Loss       Best Estimate 
-----------  ---------------
  Linear         Median     
 Quadratic        Mean      
    0/1           Mode      

When making point estimates of unknown parameters, we should make the choices that minimize the loss. Nevertheless, the best estimate depends on the kind of loss function we are using, and we will discuss in more depth how these best estimates are determined in the next section.

### Working with Loss Functions

UNFINISHED

<div class="figure" style="text-align: center">
<img src="03-decision-01-decisions_files/figure-html/posterior-decision-1.png" alt="Posterior" width="960" />
<p class="caption">(\#fig:posterior-decision)Posterior</p>
</div>

Figure \@ref(fig:posterior-decision)

Linear loss: The absolute value function is required because overestimates and underestimates do not cancel out.

L0 is minimized at the mode of the posterior distribution.
L1 is minimized at the median of the posterior distribution.
L2 is minimized at the mean of the posterior distribution.

### Minimizing Expectated Loss for Hypothesis Testing

### Posterior Probabilities of Hypotheses and Bayes Factors
