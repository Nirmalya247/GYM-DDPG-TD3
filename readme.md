# Actor-critic methods for Reinforcement Learning
## 1 Introduction and Problem Statement
In many situations the action space is a set of discrete individual actions. Like in board games.
But in physical world or physics simulation environments we need to take continuous actions.
In this project we will study actor-critic methods for continuous control space specially DDPG, how
it works, the mathematical background, and study the environments which are best suited for the
algorithm (DDPG) and best reward range for the algorithm.
## 2 Literature Review
In this section we will see how RL algorithms works starting from discrete action space to continuous
action space
## 2.1 DQN
In DQN algorithms we consider the actor can take discrete set of $K$ actions $a = (1, ...., K)$. and an
environment $E$ which gives feedback or reward $r$, for an action a in sate $s$. $Q$ is function of $a$ and $r$


DQN try to predict the total reward for all actions given a state. Then we will select the action with maximum reward. We can have a table called $Q-table$ for states and actions which will contain possible rewards.

Here the $Q-table$ is updated through **Bellman Equation**:

$Q(s_{t}, a_{t}) = (1 - \alpha) * Q(s_{t}, a_{t}) + \alpha * r_{t} + \lambda * max_{a' \in K}Q(s_{t + 1}, a'))$


Here we are updating the possible reward at for action $a$ at state $s_{t}$ with the given reward $r_{t}$ by environment at that sate and the possible maximum total reward in next state $s_{t + 1}$ for taking action $a$ at the current state.

This algorithm works fine for small action space. But for big action and state space $Q$ table will be so large that we would not be able to store.

We can use NN to approximate or learn the Q table. Where $Q$ will be a neural network which takes $state$ as input and give possible rewards for all the actions.

We will have two networks $Q$ and $Q'$. $Q$ will be updated at every iteration and $Q'$ will be updated (weights will be copied from $Q$) after each episode of the environment.

So, the new update rule for $Q$ is:

$Q_{new}(s_{t}, a_{t}) = (1 - \alpha) * Q(s_{t}, a_{t}) + \alpha * r_{t} + \lambda * max_{a' \in K}Q'(s_{t + 1}, a'))$

We will store $(s_{t}, a_{t}, r_{t}, s_{t + 1})$ in a set $R$
Next we will sample a mini batch of $N$ transitions ${(s_{t_{i}}, a_{t_{i}}, r_{t_{i}}, s_{(t + 1)_ {i}})}|_ {i = 1}^{N}$  and train $Q$ for input state $S$ and output new action $a_{new}$
## 2.2 DDPG
The above algorithm **DQN** works great in discrete action space but can not work in continuous action space and for this the paper **[1]** gives us the solution **DDPG**, an algorithm that works in continuous action space using the same **Bellman** update rule.
## 2.2.1 Problem with DQN
**DQN** tries to find the maximum rewarding action. But in a continuous action space we do not have such a flexibility. One solution could be to increase the action space in intervals. But as the output size of $Q$ network is the size of action space this will increase the complexity of network and would not be feasible. And with the increase in number of degrees of freedom. For example if we had the size of action space for one degree of freedom $100$ and number of degree of freedom is $12$ the size of the output of neural network will be $100^{12}$ which is infeasible in a neural network.
## 2.2.2 Approach of DDPG
**DDPG** solves the above problem with an \textbf{Actor Network $\mu(s_{t}|\theta^{\mu})$} which predicts the continuous action $a_{t}$ given state $s_{t}$ and we have the previous $Q$ network named as critic, defined as $Q(s_{t}, a_{t}|\theta^{Q})$ where $\theta^{\mu})$ and $\theta^{Q}$ are function approximates. (we can think of them as weights of neural network).


The critic $Q(s_{t}, a_{t}|\theta^{Q})$ can be updated with a deterministic target policy $\mu : S \to A$ using Loss function as:

$L(\theta^{Q}) = (Q(s_{t}, a_{t}|\theta^{Q}) - y_{t})^2$

Where $y_{t}$ is given by the Bellman equation as:

$y_{t} = r(s_{t}, a_{t}) + \gamma*Q(s_{t + 1}, \mu(s_{t + 1})|\theta^{Q})$



The actor $\mu(s_{t}|\theta^{\mu})$ can be updated using the sample policy gradient:

$\nabla_{\theta^{Q}}J = \frac{1}{N} * \sum_{i} \nabla_{a}Q(s, a | \theta^{Q}) | \nabla_{Q^{\mu}}\mu(s|\theta^{\mu}) | s_{i}$


Similar to DQN we use a minibatch to update $Q$ and we also use two separate target networks $Q'$ and $\mu'$ and update the weights of them using soft target update method as $Q$ is prone to diverge.

$\theta' = \tau * \theta + (1 - \tau) * \theta'$


The algorithm for **DDPG** can be found in the paper **[1]**
## 3 Solution
I have implemented DDPG algorithm and experimented with some of the popular environments provided by OpenAI and compared with TD3.


I have tried my get better result from DDPG by changing reward function and random sampling environment at the beginning.


For TD3 I have used available sources as my project is based on DDPG and it's performance.
## 4 Empirical methodology
For testing I have used 2 classic continuous environments and 3 Mujoco environments from Open-AI.

My model specifications are:
```
+----------------------------+-------------------+-------------------------------------------+
| Name                       | State and Action  | Reward Function                           |
+----------------------------+-------------------+-------------------------------------------+
| Pendulum-v1                | 3, 1[-2, +2]      | Default Function (+1 each state)          |
+----------------------------+-------------------+-------------------------------------------+
| MountainCarContinuous-v0   | 2, 1[-1, +1]      | Modified Reward Function                  |
+----------------------------+-------------------+-------------------------------------------+
| InvertedDoublePendulum-v2  | 11, 1[-1, +1]     | Default Function                          |
+----------------------------+-------------------+-------------------------------------------+
| HalfCheetah-v3             | 17, 6[-1, +1]     | Default Function                          |
+----------------------------+-------------------+-------------------------------------------+
| Ant-v3                     | 111, 8[-1, +1]    | Default Function (With some modification) |
+----------------------------+-------------------+-------------------------------------------+
```
From my observation DDPG works best if reward is continuously increasing towards best action. Which is true in Pendulum-v1 and InvertedDoublePendulum-v2 but in other environments that is not the case.


So, exploring the whole environment is tough for the Models and Critic network don't learn the environment correctly.

There are four major steps we need to take:
- Max reward need to be 0. Theoretically the algorithm does not need to have a bound on reward. But if our max reward is 0. that will be a stable region and our Actor will also explore more as new Models will give 0, so unexplored regions will be considered as maximum reward region.
- For step based environments we can have a reward function which will give same reward for a set of action at the end of the step and add that to our memory. For example in case of MountainCarContinuous-v0 we can think a oscillation as a step and reward based on the oscillation properties, next for HalfCheetah-v3 and Ant-v3 we can think a waling step as our step and reward based on how good was the walk other wise it can stop exploring at all.
- As number of action space increases, exploration becomes hard. So, we need to sample some actions in the beginning from the environment. In Ant-v3 it was very helpful.
- Eventually decreasing SD for the Normal nose based of reward is giving good result.

Initially parameters were set to:
- Normal Noise $SD$: 0.2
- $\gamma$: 0.99
- $\tau$: 0.005
- batch size: 64 for Pendulum-v1, 128 for others (based on model complexity)
- memory size: 100000-1000000 (based on exploration time)
## 5 Results
Result in different environments:
```
+---------------------------+----------------+---------------------------+
| Name                      | Episode Taken  | Last Episode Reward DDPG  |
+---------------------------+----------------+---------------------------+
| Pendulum-v1               | 100            | -1.5398104168652305       |
+---------------------------+----------------+---------------------------+
| MountainCarContinuous-v0  | 140            | 74.06378049341234         |
+---------------------------+----------------+---------------------------+
| InvertedDoublePendulum-v2 | 23235          | -10.443067281410592       |
+---------------------------+----------------+---------------------------+
| HalfCheetah-v3            | 200            | 2009.6759814067225        |
+---------------------------+----------------+---------------------------+
| Ant-v3                    | 348            | 771.5938868018742         |
+---------------------------+----------------+---------------------------+
 ```
 ## 5.1 DDPG and TD3 reward vs episode
 <table>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/Nirmalya247/GYM-DDPG-TD3/master/images/pendulum.png" style="width:400px">
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/Nirmalya247/GYM-DDPG-TD3/master/images/montain_car.png" style="width:400px">
    </td>
  </tr>
  <tr>
    <td>
      pendulum
    </td>
    <td>
      montain car
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/Nirmalya247/GYM-DDPG-TD3/master/images/half_cheeta.png" style="width:400px">
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/Nirmalya247/GYM-DDPG-TD3/master/images/ant.png" style="width:400px">
    </td>
  </tr>
  <tr>
    <td>
      half cheeta
    </td>
    <td>
      ant
    </td>
  </tr>
  <tr>
    <td colspan="2">
      <img src="https://raw.githubusercontent.com/Nirmalya247/GYM-DDPG-TD3/master/images/double_pendulum.png" style="width:400px">
    </td>
  </tr>
  <tr>
    <td colspan="2">
      double pendulum
    </td>
  </tr>
</table>

## 6 Discussion
DDPG is good but has many shortcoming in real applications (Environments with high action space or Step based environment).
- **Steps**: It does not works directly in a step based environment as discussed above (in empirical methodology)
- **Reward**: Best exploration is possible if we have maximum reward 0. But in many real environments we don't know how good step we can get in future like walking. So, we can not set a maximum reward.
- **Exploration**: Critic expects to get a full view of the environment, but exploration in large action space is hard and in step based environments it becomes harder as Adding nose also does not get to all possible steps. (For example we can not explore all type of steps in a walking environment by adding noise only) We need to random sample environment at the beginning for this.
