# CartPole-solution

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1, 0 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

*like [GYM CartPole-v1](https://gym.openai.com/envs/CartPole-v1), but may not apply force.


### example

```python
oo7 = Agent(
env=Cart_pole,
epochs=5000,
learning_rate=0.05, 
gamma=0.9,
epsilon=0.1
)

oo7.fit()
```
