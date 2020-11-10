import utils

class Trajectories(object):

    def __init__(self, s_lst_lst, a_lst_lst):
        self.initialize(s_lst_lst, a_lst_lst)

    @classmethod
    def discrete(cls, s_lst_lst, a_lst_lst):
        return Trajectory(s_lst_lst, a_lst_lst)

    @classmethod
    def discrete_compact(cls, s_lst_lst, a_lst_lst):
        return Trajectory(utils.compact_paths(s_lst_lst), a_lst_lst)

    @classmethod
    def discrete_compact_4_actions(cls, s_lst_lst, a_lst_lst):
        return Trajectory(utils.compact_paths(s_lst_lst, restrict_4_actions=True), a_lst_lst)

    @classmethod
    def continuous(cls, s_lst_lst, a_lst_lst, discretization_fn):
        return Trajectory(discretization_fn(s_lst_lst), a_lst_lst)

    @classmethod
    def continuous_compact(cls, s_lst_lst, a_lst_lst, discretization_fn):
        return Trajectory(utils.compact_paths(discretization_fn(s_lst_lst)), a_lst_lst)

    @classmethod
    def continuous_compact_4_actions(cls, s_lst_lst, a_lst_lst, discretization_fn):
        return Trajectory(utils.compact_paths(discretization_fn(s_lst_lst), restrict_4_actions=True), a_lst_lst)

    def initialize(self, s_lst_lst, a_lst_lst):
        self.s_lst_lst = s_lst_lst
        self.s_lst_lst_format_xy = [[(p[1], p[0]) for p in s_lst] for s_lst in self.s_lst_lst]
        self.a_lst_lst = a_lst_lst

    def read(self, states_only=False, s_a_zipped=False):
        if states_only:
            return self.s_lst_lst
        else:
            if s_a_zipped:
                return [list(zip(self.s_lst_lst[i], self.a_lst_lst[i])) for i in range(len(self.s_lst_lst))]
            else:
                return self.s_lst_lst, self.a_lst_lst

    def read_xy(self, states_only=False, s_a_zipped=False):
        if states_only:
            return self.s_lst_lst_format_xy
        else:
            if s_a_zipped:
                return [list(zip(self.s_lst_lst_format_xy[i], self.a_lst_lst[i])) for i in range(len(self.s_lst_lst_format_xy))]
            else:
                return self.s_lst_lst_format_xy, self.a_lst_lst
