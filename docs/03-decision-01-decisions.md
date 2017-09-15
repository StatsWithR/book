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

UNFINISHED: CURRENTLY WORKING ON L2

Now we illustrate why certain estimates minimize certain loss functions. 

\BeginKnitrBlock{example}<div class="example"><span class="example" id="ex:car"><strong>(\#ex:car)</strong></span>You work at a car dealership. Your boss wants to know how many cars the dealership will sell per month. An analyst who has worked with past data from your company provided you a distribution that shows the probability of number of cars the dealership will sell per month. In Bayesian lingo, this is called the posterior distribution. A dot plot of that posterior is shown in Figure \@ref(fig:posterior-decision). The mean, median and the mode of the distribution are also marked on the plot. Your boss doesn't know any Bayesian statistics though, so he/she wants you to report **a single number** for the number of cars the dealership will sell per month.</div>\EndKnitrBlock{example}

<div class="figure" style="text-align: center">
<img src="03-decision-01-decisions_files/figure-html/posterior-decision-1.png" alt="Posterior" width="960" />
<p class="caption">(\#fig:posterior-decision)Posterior</p>
</div>

Suppose your single guess is 30, and we call this $g$ in the following calculations. If your loss function is $L_0$ (i.e., a 0/1 loss), then you lose a point for each value in your posterior that differs from your guess and do not lose any points for values that exactly equal your guess. The total loss is the sum of the losses from each value in the posterior.

In mathematical terms, we define $L_0$ (0/1 loss) as

$$L_{0,i}(0,g) = \left\{ \begin{array}{cc}
0 & \text{if } g=x_i \\ 1 & \text{otherwise}
\end{array}\right.$$

The total loss is $L_0 = \sum_i L_{0,i}(0,g)$.

Let's calculate what the total loss would be if your guess is 30. Table \@ref(tab:L0-table) summarizes the values in the posterior distribution sorted in descending order. 

The first value is 4, which is not equal to your guess of 30, so the loss for that value is 1. The second value is 19, also not equal to your guess of 30, and the loss for that value is also 1. The third value is 20, also not equal to your guess of 30, and the loss for this value is also 1.

There is only one 30 in your posterior, and the loss for this value is 0 -- since it's equal to your guess (good news!). The remaining values in the posterior are all different than 30 hence, the loss for them are all ones as well. 

To find the total loss, we simply sum over these individual losses in the posterior distribution with 51 observations where only one of them equals our guess and the remainder are different. Hence, the total loss is 50. 

Figure \@ref(fig:L0-mode) is a visualization of the posterior distribution, along with the 0-1 loss calculated for a series of possible guesses within the range of the posterior distribution. To create this visualization of the loss function, we went through the process we described earlier for a guess of 30 for all guesses considered, and we recorded the total loss. We can see that the loss function has the lowest value when $g$, our guess, is equal to **the most frequent observation** in the posterior. Hence, $L_0$ is minimized at the **mode** of the posterior, which means that if we use the 0/1 loss, the best point estimate is the mode of the posterior. 


Table: (\#tab:L0-table)L0: 0/1 loss for g = 30

 i      x_i     L0: 0/1 
----  -------  ---------
 1       4         1    
 2      19         1    
 3      20         1    
        ...       ...   
 14     30         0    
        ...       ...   
 50     47         1    
 51     49         1    
       Total      50    

<div class="figure" style="text-align: center">
<img src="03-decision-01-decisions_files/figure-html/L0-mode-1.png" alt="L0 is minimized at the mode of the posterior" width="960" />
<p class="caption">(\#fig:L0-mode)L0 is minimized at the mode of the posterior</p>
</div>

Let's consider another loss function. If your loss function is $L_1$ (i.e., linear loss), then the total loss for a guess is the sum of the **absolute values** of the difference between that guess and each value in the posterior. Note that the absolute value function is required, because overestimates and underestimates do not cancel out.

In mathematical terms, $L_1$ (linear loss) is calculated as $L_1(g) = \sum_i |x_i - g|$.

We can once again calculate the total loss under $L_1$ if your guess is 30. Table \@ref(tab:L1-table) summarizes the values in the posterior distribution sorted in descending order.

The first value is 4, and the absolute value of the difference between 4 and 30 is 26. The second value is 19, and the absolute value of the difference between 19 and 30 is 11. The third value is 20 and the absolute value of the difference between 20 and 30 is 10. 

There is only one 30 in your posterior, and the loss for this value is 0 since it is equal to your guess. The remaining value in the posterior are all different than 30 hence their losses are different than 0. 

To find the total loss, we again simply sum over these individual losses, and the total is to 346. 

Again, Figure \@ref(fig:L1-median) is a visualization of the posterior distribution, along with a linear loss function calculated for a series of possible guesses within the range of the posterior distribution. To create this visualization of the loss function, we went through the same process we described earlier for all of the guesses considered. This time, the function has the lowest value when $g$ is equal to the **median** of the posterior. Hence, $L_1$ is minimized at the **median** of the posterior one other loss function. 


Table: (\#tab:L1-table)L1: linear loss for g = 30

 i      x_i     L1: |x_i-30| 
----  -------  --------------
 1       4           1       
 2      19           1       
 3      20           1       
        ...         ...      
 14     30           0       
        ...         ...      
 50     47           1       
 51     49           1       
       Total        346      

<div class="figure" style="text-align: center">
<img src="03-decision-01-decisions_files/figure-html/L1-median-1.png" alt="L1 is minimized at the median of the posterior" width="960" />
<p class="caption">(\#fig:L1-median)L1 is minimized at the median of the posterior</p>
</div>

UNFINISHED

If your loss function is L2, that is a squared loss, then the total loss for a guess is the sum of the squared differences between that guess and each value in the posterior. We can once again calculate the total loss under L2 if your guess is 30. We have the posterior distribution again, sorted in ascending order. The first value is 4, and the squared difference between 4 and 30 is 676. The second value is 19 the square of the difference between 19 and 30 is 121. The third value is 20, and the square difference between 20 and 30 is 100. There's only 1 30 in your posterior, and the loss for this value is 0 since it's equal to your guess. The remaining values in the posterior are again all different than 30, hence their losses are all different than 0. To find the total loss, we simply sum over these individual losses again and the total loss comes out to 3,732. We have the visualization of the posterior distribution. Again, this time along with the squared loss function calculated for a possible serious of possible guesses within the range of the posterior distribution. Creating this visualization had the same steps. Go through the same process described earlier for a guess of 30, for all guesses considered, and record the total loss. This time, the function has the lowest value when X is equal to the mean of the posterior. Hence, L2 is minimized at the mean of the posterior distribution. In summary, in this lesson we illustrated that the 0, 1 loss, L0 is minimized at the mode of the posterior distribution. The linear loss L1 is minimized at the median of the posterior distribution and the squared loss L2 is minimized at the mean of the posterior distribution. Going back to the original question. The point estimate to report to your boss about the number of cars the dealership will sell per month depends on your loss function. In any case, you would choose to report the estimate that minimizes the loss. 


Table: (\#tab:L2-table)L2: squared loss for g = 30

 i      x_i     L2: (x_i-30)^2 
----  -------  ----------------
 1       4            1        
 2      19            1        
 3      20            1        
        ...          ...       
 14     30            0        
        ...          ...       
 50     47            1        
 51     49            1        
       Total         3732      

<div class="figure" style="text-align: center">
<img src="03-decision-01-decisions_files/figure-html/L2-mean-1.png" alt="L2 is minimized at the mean of the posterior" width="960" />
<p class="caption">(\#fig:L2-mean)L2 is minimized at the mean of the posterior</p>
</div>


L0 is minimized at the mode of the posterior distribution.
L1 is minimized at the median of the posterior distribution.
L2 is minimized at the mean of the posterior distribution.

### Minimizing Expectated Loss for Hypothesis Testing

### Posterior Probabilities of Hypotheses and Bayes Factors
