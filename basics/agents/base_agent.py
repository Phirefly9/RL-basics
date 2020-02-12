class Agent():
    def select_action(self, state):
        raise NotImplementedError

    def finish_iteration(self, reward, iteration_count):
        raise NotImplementedError

    def finish_episode(self, episode):
        raise NotImplementedError

    def check_solved_env(self, reward_threshold):
        raise NotImplementedError
    
    @staticmethod
    def add_args(parser):
        raise NotImplementedError

