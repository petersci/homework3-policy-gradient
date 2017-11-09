
# 江愷笙 <span style="color:black">(106062568)</span>

# Homework3-Policy-Gradient report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

## Overview

>Policy gradient methods are a type of reinforcement learning techniques that rely upon optimizing parametrized policies with respect to the expected return (long-term cumulative reward) by gradient descent. They do not suffer from many of the problems that have been marring traditional reinforcement learning approaches such as the lack of guarantees of a value function, the intractability problem resulting from uncertain state information and the complexity arising from continuous states & actions.
## Implementation

In this homework we have to use policy gradient method to solve the cartpole problem.
* Problem 1

In problem 1 we have to use tensorflow to construct DQN for the policy gradient. Here we have to add the softmax layer to obtain the probability distribution.

```python
fc = tf.layers.dense(self._observations, hidden_dim, tf.tanh)
probs = tf.layers.dense(fc, out_dim, tf.nn.softmax)
```

* Problem 2

In problem 2 we have to define our lost for the neural network, here the loss function is surrogate loss. However, we have to take care that for the optimizer in tensorflow, the task is to minimize the loss (gradient descent), but in policy gradient, we have to maximize the surrogate loss (gradient asscent). So here we take the negative number of the loss to minimize this negative number, which equals to maximize its positive number.

<tr>
<td>
<img src="imgs/surrogate_loss.PNG"/>
</td>
</tr>

```python
surr_loss = tf.reduce_mean(-log_prob * self._advantages)
```

the surrogate loss should take the average number over N episode and each time step, so we use tf.reduce_mean to obtain the average number.

* Problem 3

Here in problem 3 we use baseline to reduce the variance. So we replace the loss function by the formula shown below.

<tr>
<td>
<img src="imgs/baseline.PNG"/>
<img src="imgs/baseline2.PNG"/>
</td>
</tr>

```python
a = r - b
```

## Installation
* Anaconda
* Ipython notebook
* Python3.5
* OpenAI gym
* Tensorflow
* to run the code, open Lab3-policy-gradient.ipynb by using Ipython notebook and execute each block.

## Results
* Value Iteration

Here we can see the action and value update for each state.

<tr>
<td>
<img src="imgs/VI_1.PNG" width="19%"/>
<img src="imgs/VI_2.PNG" width="19%"/>
<img src="imgs/VI_3.PNG" width="19%"/>
<img src="imgs/VI_4.PNG" width="19%"/>
<img src="imgs/VI_5.PNG" width="19%"/>
<img src="imgs/VI_6.PNG" width="19%"/>
<img src="imgs/VI_7.PNG" width="19%"/>
<img src="imgs/VI_8.PNG" width="19%"/>
<img src="imgs/VI_9.PNG" width="19%"/>
<img src="imgs/VI_10.PNG" width="19%"/>
<img src="imgs/VI_plot.PNG"/>
</td>
</tr>

* Policy Iteration

<tr>
<td>
<img src="imgs/PI_plot.PNG"/>
</td>
</tr>

* Tabular Q-learning

By finishing the tabular Q-learning, we can see a crawling robot moving fast from the left to the right.

<tr>
<td>
<img src="imgs/Q_1.PNG"/>
<img src="imgs/Q_2.PNG"/>
</td>
</tr>
