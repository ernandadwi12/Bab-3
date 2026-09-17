# Example 3.23 OpenAI Gym CartPole
# https://gym.openai.com/docs/
# https://gym.openai.com/envs/CartPole-v0/
#
# NOTE: The original 'gym' package is unmaintained and breaks with modern
# NumPy. This version uses Gymnasium (the maintained drop-in replacement)
# and its updated API:
#   - env.reset() returns (observation, info)
#   - env.step() returns (observation, reward, terminated, truncated, info)
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
for i_episode in range(20):
    observation, info = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()