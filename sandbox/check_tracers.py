import gymnasium as gym

from cardio_rl.tracers.trajectory import TrajectoryTracer

env = gym.make("CartPole-v1")

tracer = TrajectoryTracer(seq_len=5, period=1, overlapping=True)

for i in range(2):
    s, _ = env.reset()
    R = 0.0
    while True:
        a = env.action_space.sample()
        s_p, r, t, d, _ = env.step(a)
        R += r
        d = d or t
        transition = {"s": s, "a": a, "r": r, "s_p": s_p, "d": d}
        tracer.append(transition)
        print(tracer.ready)
        if tracer.ready:
            tracer.pop()
        s = s_p
        if d:
            print(f"Reward: {R}")
            break