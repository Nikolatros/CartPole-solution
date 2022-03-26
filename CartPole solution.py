import numpy as np


class Cart_pole():
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.max_grad = 15 * 2 * np.pi / 360
        self.max_x = 2.4
        
    def render(self):
        """ Generates first environment state

        Returns:
            _numpy.ndarray_: first state params
        """        
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state

    def get_state(self, f):
        """ return new state params, reward, done
        reward = 1 if Pole didn't fall else 0 
        done  = 0 if Pole didn't fall else 1

        Args:
            f (_int_): force applied to cart

        Returns:
            _numpy.ndarray_: new state params
            _int_: the reward  
            _bool_: completion indicator
        """        
        o, x, o_u, x_u = self.state
        
        dt = self.tau
        temp = (f + self.polemass_length*o_u**2*np.sin(o))/self.total_mass
        
        o_w = (self.gravity*np.sin(o) - np.cos(o)*temp)/(self.length*(0.75 - self.masspole*np.cos(o)**2/self.total_mass))
        o_u += o_w*dt
        o += o_u*dt

        x_w = temp - self.polemass_length*o_w*np.cos(o)/self.total_mass
        x_u += x_w*dt
        x += x_u*dt

        self.state = [o, x, o_u, x_u]
        
        if np.abs(o) > self.max_grad or np.abs(x) > self.max_x:
            done = 1
            reward = 0
        else:
            reward = 1
            done = 0

        return self.state, reward, done

class Agent():
    def __init__(self, env, epochs, learning_rate, gamma, epsilon):
        self.env = env()
        self.strategy = dict()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gamma = gamma # discount coefficient
        self.epsilon = epsilon # probability of explore
        self.ticks_line = []

    def state_processing(self, state):
        """Binarizes environment params to write in strategy

        Args:
            state (_numpy.ndarray_): current state params

        Returns:
            _turple_: processed state params
        """        
        bins = [np.arange(-0.26,0.27,0.02), np.arange(-2.4,2.5,0.2), np.arange(-10,11,1), np.arange(-10,11,1)]
        processed_state = tuple()
        for i in range(4):
            processed_state += ((np.digitize(state[i], bins[i])),)
        return processed_state

    def make_action(self, processed_state):
        """Selects an action for epsilon-greedy algorithm

        Args:
            processed_state (_numpy.ndarray_): current precessed state params

        Returns:
            _int_: index of action
        """        
        Q_state = self.strategy[processed_state]
        if np.random.random() < self.epsilon:
            action_index = np.random.choice(3)
        else:
            action_index = Q_state.index(np.max(Q_state))
        return action_index

    def estimate_progress(self, n):
        """Return data for plotting with average value for the last N ticks

        Args:
            n (_int_): number of last values
            
        Returns:
            _list_: data for plotting
        """        
        stat = []
        for i in range(len(self.ticks_line)):
            if i >= n-1:
                stat.append(np.sum(self.ticks_line[i-n+1:i+1])/n)
        return [list(range(len(stat))), stat]

    def add_state_if_missing(self, processed_state):
        """Generate Q-values nulls if absent in the strategy

        Args:
            processed_state (_numpy.ndarray_): current processed state params
        """        
        try:
            self.strategy[processed_state]
        except KeyError:
            self.strategy[processed_state] = [0, 0, 0]

    def fit(self):
        """Creates and improves the action policy
        """
        progress = 0 # for future visualization
        for epoch in range(self.epochs):
            state = self.env.render()
            processed_state = self.state_processing(state)
            self.add_state_if_missing(processed_state)
            epsilon = self.epsilon
            done = 0
            ticks = 0
            while (not done) and (ticks<500):
                # Get action index
                action_index = self.make_action(processed_state)
                f = action_index - 1
                # Get state params
                new_state, reward, done = self.env.get_state(f)
                processed_new_state = self.state_processing(new_state)
                # Update strategy
                self.add_state_if_missing(processed_new_state)
                max_next_q = np.max(self.strategy[processed_new_state])
                error = reward + self.gamma*max_next_q - self.strategy[processed_state][action_index]
                self.strategy[processed_state][action_index] += self.learning_rate * error
                # Update state
                processed_state = processed_new_state
                # Update learn perams
                ticks += 1
                if ticks % 10 == 0:
                    epsilon *= 0.99
            else:
                self.ticks_line.append(ticks)
                if (epoch + 1) % (self.epochs/10) == 0:
                    progress += 1
                    print(progress, end='/10 ')
