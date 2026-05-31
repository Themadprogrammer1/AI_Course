import ext_elev
import numpy as np

id = ["000000000"]


class Controller:
    """Stochastic multi-elevator controller.

    Implement choose_next_action(state) to return a single legal action
    string. See the assignment PDF (Section 5) for the full API contract
    and the engine-access policy.
    """

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.reachable = self.game.get_reachable()
        self.capacities = self.game.get_capacities()
        self.horizon = self.game.get_max_steps();
        self.goalReward = self.game.get_goal_reward();
        
        _, persons_t, _ = self.game.get_initial_state()
        person_ids = [pid for pid, _ in persons_t]
        
        self.e_prob = {eid: self.game.get_elevator_action_prob(eid) for eid in self.reachable}
        self.p_prob = {pid: self.game.get_person_action_prob(pid) for pid in person_ids}
        self.p_weight = {pid: self.game.get_person_weight(pid) for pid in person_ids}
        self.P_goal = {pid: self.game.get_person_goal(pid) for pid in person_ids}
        self.p_reward = {pid: self.game.get_person_reward(pid) for pid in person_ids}
        self.p_avg_reward = {pid: sum(self.p_reward[pid]) / len(self.p_reward[pid]) for pid in person_ids}


    def choose_next_action(self, state):
        elevators_t, persons_t, _ = state

        actions = ["RESET"]

        # MOVE{e, f}: f in reachable[e], f != current floor
        for (eid, cur_f, _) in elevators_t:
            for f in self.reachable[eid]:
                if f != cur_f:
                    actions.append(f"MOVE{{{eid},{f}}}")

        # ENTER{p, e}: person on same floor as elevator, capacity not exceeded
        for (pid, loc) in persons_t:
            if not (isinstance(loc, tuple) and loc[0] == 'floor'):
                continue
            f_p = loc[1]
            w_p = self.game.get_person_weight(pid)
            for (eid, cur_f, cur_w) in elevators_t:
                if cur_f != f_p:
                    continue
                if cur_w + w_p > self.capacities[eid]:
                    continue
                actions.append(f"ENTER{{{pid},{eid}}}")

        # EXIT{p, e}: person currently inside elevator e
        for (pid, loc) in persons_t:
            if not (isinstance(loc, tuple) and loc[0] == 'in'):
                continue
            eid = loc[1]
            actions.append(f"EXIT{{{pid},{eid}}}")

        return str(np.random.choice(actions))
