'''
basic reinforcement learning agent framework for learning different RL algorithms
'''
import gym
import argparse
import logging
from itertools import count
from basics.agents.reinforce import ReinforceAgent

AGENTS = {"reinforce": ReinforceAgent}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="environment to run")
    parser.add_argument("--render", default=False, action="store_true")
    sub_parser = parser.add_subparsers(help='sub-command help')
    for agent_name, agent_class in AGENTS.items():
        agent_sub_parser = sub_parser.add_parser(agent_name, help=f"{agent_name} help")
        agent_sub_parser = agent_class.add_args(agent_sub_parser)
        agent_sub_parser.set_defaults(agent=agent_name)
    args = parser.parse_args()

    agent_name = args.agent
    env = gym.make(args.env)
    agent = AGENTS[agent_name](args, env.observation_space.shape, env.action_space.n)


    observation = env.reset()
    for n_episodes in count(1):
        observation = env.reset()
        for n_iter in range(1, 10000):
            if args.render:
                env.render()
            action = agent.select_action(observation)
            observation, itr_reward, done, _ = env.step(action)
            agent.finish_iteration(itr_reward, n_iter)
            if done:
                break
            
        if agent.check_solved_env(env.spec.reward_threshold):
            print("Solved env, stopping at episode", n_episodes, "iter", n_iter)
            break
        agent.finish_episode(n_episodes)
    env.close()


if __name__ == "__main__":
    main()