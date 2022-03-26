# CartPole-solution

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1, 0 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

*like [GYM CartPole-v1](https://gym.openai.com/envs/CartPole-v1), but may not apply force.*


### example

```python
from CPS import Agent, Cart_pole
# Create the environment
env = Cart_pole()
# Create the agent
oo7 = Agent(env=env, epochs=10000, learning_rate=0.05, gamma=0.9, epsilon=0.1)
# Let's train this
oo7.fit()
# Now can see how we did.
oo7.play()
# output:
# The Agent ended the game at step 69.
```

### If you have *matplotlib* you can use the method *estimate_porgress*
```python
import matplotlib.pyplot as plt
x, y = oo7.estimate_progress(1000)
plt.plot(x, y)

```

**or**
```python
plt.plot(*oo7.estimate_progress(1000))

```
**And get**


Average duration of the game for 1000 previous attempts for all data


![image](https://user-images.githubusercontent.com/98982329/160242305-f928fd10-18b3-4d2f-8355-7943d0218b1c.png)
