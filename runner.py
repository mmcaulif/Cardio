from collections import deque

class Runner():
    def __init__(
            self,
            env,
            length,
            flush,
            sampler,
            capacity,
            trajectory,
            freq
        ) -> None:

        self.env = env
        self.length = length
        self.flush = flush
        self.sampler = sampler
        self.capacity = capacity
        self.trajectory = trajectory
        self.freq = freq      
        self.train_after = 1000
        self.buffer = self.flush_buffer()

        # environment init (move to external function)
        self.state, _ = self._reset()
        self.warm_start()

        pass

    def flush_buffer(
        self
        ):
        self.buffer = deque(maxlen=self.capacity)
        return self.buffer

    def warm_start(
        self
    ):
        for _ in range(self.train_after):
            a = self.env.action_space.sample()
            s_p, r, d, t, info = self._step(a)
            self.buffer.append([self.state, a, r, s_p, d])
            self.state = s_p
            if d or t:
                self.state, _ = self._reset()

        # return list(self.buffer)
        pass

    def _step(
            self,
            a):
        return self.env.step(a)

    def _reset(
        self
    ):
        return self.env.reset()
    
    def agent_step(
        self,
        policy,
        state=None
    ):
        if state:
            pass

        else:
            state = self.state        

        if policy == 'random':
            return self.env.action_space.sample()

    def get_batch(
            self,
            net,
            policy
            ):
        
        self.net = net

        for _ in range(self.length):
            a = self.agent_step(policy)
            s_p, r, d, t, info = self._step(a)
            self.buffer.append([self.state, a, r, s_p, d])
            self.state = s_p
            if d or t:
                self.state, _ = self._reset()

        batch = self.create_batch()

        return self.prep_batch(batch)

    def create_batch(
        self,
        ):

        """
        either samples or just copies over the batch, handle flushing and sampling here
        """

        batch = list(self.buffer)

        if self.flush:
            self.flush_buffer()

        return batch

    def prep_batch(
        self,
        batch
        ):

        """
        takes the batch (which will be a list of transitons) and processes them to be seperate etc.
        """

        return batch    # try return it as dict maybe? or just something fancier and seperated