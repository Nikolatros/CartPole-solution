# CartPole-solution

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1, 0 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, the cart moves more than 2.4 units from the center or the number of steps in the game is more than specified maximum number of steps.

*like [GYM CartPole-v1](https://gym.openai.com/envs/CartPole-v1), but may not apply force and number of steps are limited.*


### example

```python
from CPS import Agent, Cart_pole
np.random.seed(43)
# Create the environment
env = Cart_pole()
# Create the agent
oo7 = Agent(env=env, epochs=10000, learning_rate=0.05, gamma=0.9, epsilon=0.1, max_ticks=500)
# Let's train this
oo7.fit()
# Now can see how we did.
oo7.play()
# output:  The Agent ended the game at step 51.
# let's see how it developed.

oo7.estimate_progress(1000)

```
**output: plot with average duration of the game for 1000 previous attempts for all games**

![image](https://user-images.githubusercontent.com/98982329/160244866-44be173a-84d5-4b2c-b8f6-d2dcd7af069c.png)
