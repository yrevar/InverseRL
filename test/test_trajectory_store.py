from unittest import TestCase
from irl.data import TrajectoryStore

class TestTrajectoryStore(TestCase):
    def test(self):
        s_lst_1, a_lst_1 = [(1, 1), (2, 2), (3, 3)], ['A', 'B', None]
        s_lst_2, a_lst_2 = [(4, 1), (4, 2), (4, 3)], ['C', 'D', None]
        ts = TrajectoryStore([s_lst_1, s_lst_2], [a_lst_1, a_lst_2])
        self.assertEqual(len(ts), 2)
        self.assertEqual(ts.read_states(), [s_lst_1, s_lst_2])
        self.assertEqual(ts.read_states_xy(), [ts._s_lst_to_xy_format(s_lst_1), ts._s_lst_to_xy_format(s_lst_2)])
        self.assertEqual(list(ts.read(zip_sa=False)), list(zip([s_lst_1, s_lst_2], [a_lst_1, a_lst_2])))
        self.assertEqual(list(ts.read(zip_sa=True))[0], [((1, 1), 'A'), ((2, 2), 'B'), ((3, 3), None)])
        self.assertEqual(list(ts.read(zip_sa=True))[1], [((4, 1), 'C'), ((4, 2), 'D'), ((4, 3), None)])
        self.assertEqual(list(ts.read_xy())[0], (ts._s_lst_to_xy_format(s_lst_1), a_lst_1))
        self.assertEqual(list(ts.read_xy())[1], (ts._s_lst_to_xy_format(s_lst_2), a_lst_2))