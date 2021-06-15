import utils
from collections import defaultdict

class TrajectoryStore(object):

    def __init__(self, s_lst_lst, a_lst_lst):
        self.initialize(s_lst_lst, a_lst_lst)

    @classmethod
    def discrete(cls, s_lst_lst, a_lst_lst):
        return TrajectoryStore(s_lst_lst, a_lst_lst)

    @classmethod
    def discrete_compact(cls, s_lst_lst, a_lst_lst):
        return TrajectoryStore(utils.compact_paths(s_lst_lst), a_lst_lst)

    @classmethod
    def discrete_compact_4_actions(cls, s_lst_lst, a_lst_lst):
        return TrajectoryStore(utils.compact_paths(s_lst_lst, restrict_4_actions=True), a_lst_lst)

    @classmethod
    def continuous(cls, s_lst_lst, a_lst_lst, discretization_fn):
        return TrajectoryStore(discretization_fn(s_lst_lst), a_lst_lst)

    @classmethod
    def continuous_compact(cls, s_lst_lst, a_lst_lst, discretization_fn):
        return TrajectoryStore(utils.compact_paths(discretization_fn(s_lst_lst)), a_lst_lst)

    @classmethod
    def continuous_compact_4_actions(cls, s_lst_lst, a_lst_lst, discretization_fn):
        return TrajectoryStore(utils.compact_paths(discretization_fn(s_lst_lst), restrict_4_actions=True), a_lst_lst)

    def _s_lst_to_xy_format(self, s_lst):
        return [(p[1], p[0]) for p in s_lst]

    def _s_lst_lst_to_xy_format(self, s_lst_lst):
        return [self._s_lst_to_xy_format(s_lst) for s_lst in self.s_lst_lst]

    def initialize(self, s_lst_lst, a_lst_lst):
        self.s_lst_lst = s_lst_lst
        self.a_lst_lst = a_lst_lst
        self.goal_to_s_lst_lst = defaultdict(lambda: [])
        self.goal_to_a_lst_lst = defaultdict(lambda: [])
        for idx, s_lst in enumerate(self.s_lst_lst):
            self.goal_to_s_lst_lst[self.compute_goal(s_lst)].append(s_lst)
            if a_lst_lst is not None:
                self.goal_to_a_lst_lst[self.compute_goal(s_lst)].append(self.a_lst_lst[idx])

        self.goals = list(self.goal_to_s_lst_lst.keys())

    def infer_actions(self, infer_fn=lambda s_lst_lst: []):
        self.a_lst_lst = infer_fn(self.s_lst_lst)
        self.initialize(self.s_lst_lst, self.a_lst_lst)

    def __read_states(self, format="yx"):
        if format == "yx":
            return self.s_lst_lst
        else:
            return self._s_lst_lst_to_xy_format(self.s_lst_lst)

    def read_states(self):
        return self.__read_states("yx")

    def read_states_xy(self):
        return self.__read_states("xy")

    def read(self, zip_sa=False):
        s_lst_lst = self.__read_states(format="yx")
        a_lst_lst = self.a_lst_lst
        for i in range(len(s_lst_lst)):
            if zip_sa:
                yield list(zip(s_lst_lst[i], a_lst_lst[i]))
            else:
                yield s_lst_lst[i], a_lst_lst[i]

    def read_xy(self, zip_sa=False):
        s_lst_lst = self.__read_states(format="xy")
        a_lst_lst = self.a_lst_lst
        for i in range(len(s_lst_lst)):
            if zip_sa:
                yield list(zip(s_lst_lst[i], a_lst_lst[i]))
            else:
                yield s_lst_lst[i], a_lst_lst[i]

    def read_by_goals(self, zip_sa=False):
        for goal in self.goals:
            if zip_sa:
                yield goal, [list(zip(self.goal_to_s_lst_lst[goal][i], self.goal_to_a_lst_lst[goal][i])) \
                        for i in range(len(self.goal_to_s_lst_lst[goal]))]
            else:
                yield goal, (self.goal_to_s_lst_lst[goal], self.goal_to_a_lst_lst[goal])

    def compute_goal(self, s_lst):
        return s_lst[-1]

    def get_goals(self):
        return self.goals

    def __len__(self):
        return len(self.s_lst_lst)
