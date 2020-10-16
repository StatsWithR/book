## Working with Loss Functions

Now we illustrate why certain estimates minimize certain loss functions.

\BeginKnitrBlock{example}
<span class="example" id="exm:car"><strong>(\#exm:car) </strong></span>You work at a car dealership. Your boss wants to know how many cars the dealership will sell per month. An analyst who has worked with past data from your company provided you a distribution that shows the probability of number of cars the dealership will sell per month. In Bayesian lingo, this is called the posterior distribution. A dot plot of that posterior is shown in Figure \@ref(fig:posterior-decision). The mean, median and the mode of the distribution are also marked on the plot. Your boss does not know any Bayesian statistics though, so he/she wants you to report **a single number** for the number of cars the dealership will sell per month.
\EndKnitrBlock{example}

\begin{figure}

{\centering \includegraphics{03-decision-03-working_files/figure-latex/posterior-decision-1} 

}

\caption{Posterior}(\#fig:posterior-decision)
\end{figure}

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

\begin{table}

\caption{(\#tab:L0-table)L0: 0/1 loss for g = 30}
\centering
\begin{tabular}[t]{ccc}
\toprule
i & x\_i & L0: 0/1\\
\midrule
1 & 4 & 1\\
2 & 19 & 1\\
3 & 20 & 1\\
 & ... & ...\\
14 & 30 & 0\\
\addlinespace
 & ... & ...\\
50 & 47 & 1\\
51 & 49 & 1\\
 & Total & 50\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}

{\centering \includegraphics{03-decision-03-working_files/figure-latex/L0-mode-1} 

}

\caption{L0 is minimized at the mode of the posterior}(\#fig:L0-mode)
\end{figure}

Let's consider another loss function. If your loss function is $L_1$ (i.e., linear loss), then the total loss for a guess is the sum of the **absolute values** of the difference between that guess and each value in the posterior. Note that the absolute value function is required, because overestimates and underestimates do not cancel out.

In mathematical terms, $L_1$ (linear loss) is calculated as $L_1(g) = \sum_i |x_i - g|$.

We can once again calculate the total loss under $L_1$ if your guess is 30. Table \@ref(tab:L1-table) summarizes the values in the posterior distribution sorted in descending order.

The first value is 4, and the absolute value of the difference between 4 and 30 is 26. The second value is 19, and the absolute value of the difference between 19 and 30 is 11. The third value is 20 and the absolute value of the difference between 20 and 30 is 10.

There is only one 30 in your posterior, and the loss for this value is 0 since it is equal to your guess. The remaining value in the posterior are all different than 30 hence their losses are different than 0.

To find the total loss, we again simply sum over these individual losses, and the total is to 346.

Again, Figure \@ref(fig:L1-median) is a visualization of the posterior distribution, along with a linear loss function calculated for a series of possible guesses within the range of the posterior distribution. To create this visualization of the loss function, we went through the same process we described earlier for all of the guesses considered. This time, the function has the lowest value when $g$ is equal to the **median** of the posterior. Hence, $L_1$ is minimized at the **median** of the posterior one other loss function.

\begin{table}

\caption{(\#tab:L1-table)L1: linear loss for g = 30}
\centering
\begin{tabular}[t]{ccc}
\toprule
i & x\_i & L1: |x\_i-30|\\
\midrule
1 & 4 & 26\\
2 & 19 & 11\\
3 & 20 & 10\\
 & ... & ...\\
14 & 30 & 0\\
\addlinespace
 & ... & ...\\
50 & 47 & 17\\
51 & 49 & 19\\
 & Total & 346\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}

{\centering \includegraphics{03-decision-03-working_files/figure-latex/L1-median-1} 

}

\caption{L1 is minimized at the median of the posterior}(\#fig:L1-median)
\end{figure}

If your loss function is $L_2$ (i.e. a squared loss), then the total loss for a guess is the sum of the squared differences between that guess and each value in the posterior.

We can once again calculate the total loss under $L_2$ if your guess is 30. Table \@ref(tab:L2-table) summarizes the posterior distribution again, sorted in ascending order.

The first value is 4, and the squared difference between 4 and 30 is 676. The second value is 19, and the square of the difference between 19 and 30 is 121. The third value is 20, and the square difference between 20 and 30 is 100.

There is only one 30 in your posterior, and the loss for this value is 0 since it is equal to your guess. The remaining values in the posterior are again all different than 30, hence their losses are all different than 0.

To find the total loss, we simply sum over these individual losses again and the total loss comes out to 3,732. We have the visualization of the posterior distribution. Again, this time along with the squared loss function calculated for a possible serious of possible guesses within the range of the posterior distribution.

Creating the visualization in Figure \@ref(fig:L2-mean) had the same steps. Go through the same process described earlier for a guess of 30, for all guesses considered, and record the total loss. This time, the function has the lowest value when $g$ is equal to the **mean** of the posterior. Hence, $L_2$ is minimized at the **mean** of the posterior distribution.

\begin{table}

\caption{(\#tab:L2-table)L2: squared loss for g = 30}
\centering
\begin{tabular}[t]{ccc}
\toprule
i & x\_i & L2: (x\_i-30)\textasciicircum{}2\\
\midrule
1 & 4 & 676\\
2 & 19 & 121\\
3 & 20 & 100\\
 & ... & ...\\
14 & 30 & 0\\
\addlinespace
 & ... & ...\\
50 & 47 & 289\\
51 & 49 & 361\\
 & Total & 3732\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}

{\centering \includegraphics{03-decision-03-working_files/figure-latex/L2-mean-1} 

}

\caption{L2 is minimized at the mean of the posterior}(\#fig:L2-mean)
\end{figure}

To sum up, the point estimate to report to your boss about the number of cars the dealership will sell per month **depends on your loss function**. In any case, you will choose to report the estimate that minimizes the loss.

* $L_0$ is minimized at the **mode** of the posterior distribution.
* $L_1$ is minimized at the **median** of the posterior distribution.
* $L_2$ is minimized at the **mean** of the posterior distribution.
