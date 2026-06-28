import ext_elev
import math
import heapq

id = ["000000000"]


class Controller:
    """Reinforcement-learning multi-elevator controller."""

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.capacities = game.get_capacities()
        self.reachable = game.get_reachable()
        self.sorted_e_ids = sorted(self.capacities.keys())
        
        # Factored Model
        self.move_success = {e: 0.0 for e in self.capacities}
        self.move_attempts = {e: 0.0 for e in self.capacities}
        
        initial_state = game.get_initial_state()
        self.all_persons = [p_id for p_id, loc in initial_state[1]]
        
        self.ee_success = {p: 0.0 for p in self.all_persons}
        self.ee_attempts = {p: 0.0 for p in self.all_persons}
        
        # Rewards (UCB)
        self.reward_sum = {p: 0.0 for p in self.all_persons}
        self.deliveries = {p: 0 for p in self.all_persons}
        self.total_deliveries = 0
        
        self.last_action = None
        self.last_state = None
        
        # Hyperparameters
        self.c_ucb = 1

    def get_move_prob(self, e):
        return (self.move_success[e] + 1.0) / (self.move_attempts[e] + 2.0)
        
    def get_ee_prob(self, p):
        return (self.ee_success[p] + 1.0) / (self.ee_attempts[p] + 2.0)
        
    def get_ucb_reward(self, p):
        if self.deliveries[p] == 0:
            return 1e9 # Favor unexplored people heavily
        avg = self.reward_sum[p] / self.deliveries[p]
        bonus = self.c_ucb * math.sqrt(math.log(self.total_deliveries + 1) / self.deliveries[p])
        return avg + bonus

    def get_elevator_floor(self, state, e_id):
        for e, f, w in state[0]:
            if e == e_id:
                return f
        return None
        
    def is_person_in_elevator(self, state, p_id, e_id):
        for p, loc in state[1]:
            if p == p_id:
                return loc[0] == 'in' and loc[1] == e_id
        return False
        
    def get_person_loc(self, state, p_id):
        for p, loc in state[1]:
            if p == p_id:
                return loc
        return None

    def a_star(self, state, target_p):
        start_p_loc = self.get_person_loc(state, target_p)
        if start_p_loc is None:
            return [], float('inf')
            
        e_info = {e: (f, w) for e, f, w in state[0]}
        start_e_locs = tuple(e_info[e][0] for e in self.sorted_e_ids)
        start_e_weights = tuple(e_info[e][1] for e in self.sorted_e_ids)
        
        start_node = (start_p_loc, start_e_locs, start_e_weights)
        target_goal = self.game.get_person_goal(target_p)
        target_weight = self.game.get_person_weight(target_p)
        
        pq = []
        heapq.heappush(pq, (0, 0, start_node, []))
        visited = set()
        
        while pq:
            f, g, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            
            p_loc, e_locs, e_weights = node
            
            if p_loc[0] == 'floor' and p_loc[1] == target_goal:
                return path, g
                
            # 1. Elevators MOVE
            for i, e in enumerate(self.sorted_e_ids):
                e_f = e_locs[i]
                for nxt_f in self.reachable[e]:
                    if nxt_f != e_f:
                        new_e_locs = list(e_locs)
                        new_e_locs[i] = nxt_f
                        new_node = (p_loc, tuple(new_e_locs), e_weights)
                        if new_node not in visited:
                            cost = 1.0 / self.get_move_prob(e)
                            new_g = g + cost
                            h = self._heuristic(new_node, target_p, target_goal)
                            heapq.heappush(pq, (new_g + h, new_g, new_node, path + [("MOVE", e, nxt_f)]))
                            
            # 2. Person ENTER
            if p_loc[0] == 'floor':
                for i, e in enumerate(self.sorted_e_ids):
                    if e_locs[i] == p_loc[1] and e_weights[i] + target_weight <= self.capacities[e]:
                        new_e_weights = list(e_weights)
                        new_e_weights[i] += target_weight
                        new_node = (('in', e), e_locs, tuple(new_e_weights))
                        if new_node not in visited:
                            cost = 1.0 / self.get_ee_prob(target_p)
                            new_g = g + cost
                            h = self._heuristic(new_node, target_p, target_goal)
                            heapq.heappush(pq, (new_g + h, new_g, new_node, path + [("ENTER", target_p, e)]))
                            
            # 3. Person EXIT
            if p_loc[0] == 'in':
                e = p_loc[1]
                i = self.sorted_e_ids.index(e)
                new_e_weights = list(e_weights)
                new_e_weights[i] -= target_weight
                new_node = (('floor', e_locs[i]), e_locs, tuple(new_e_weights))
                if new_node not in visited:
                    cost = 1.0 / self.get_ee_prob(target_p)
                    new_g = g + cost
                    h = self._heuristic(new_node, target_p, target_goal)
                    heapq.heappush(pq, (new_g + h, new_g, new_node, path + [("EXIT", target_p, e)]))
                    
        return [], float('inf')
        
    def _heuristic(self, node, target_p, target_goal):
        p_loc, e_locs, _ = node
        if p_loc[0] == 'floor' and p_loc[1] == target_goal:
            return 0.0
        if p_loc[0] == 'in':
            e = p_loc[1]
            i = self.sorted_e_ids.index(e)
            if e_locs[i] == target_goal:
                return 1.0 / self.get_ee_prob(target_p)
            return 1.0 / self.get_ee_prob(target_p) + 1.0 / self.get_move_prob(e)
        return 2.0 / self.get_ee_prob(target_p)

    def choose_next_action(self, state):
        if self.last_action is not None and self.last_action[0] != "RESET":
            a_type = self.last_action[0]
            if a_type == "MOVE":
                e_id = self.last_action[1]
                target_f = self.last_action[2]
                self.move_attempts[e_id] += 1
                if self.get_elevator_floor(state, e_id) == target_f:
                    self.move_success[e_id] += 1
            elif a_type == "ENTER":
                p_id = self.last_action[1]
                e_id = self.last_action[2]
                self.ee_attempts[p_id] += 1
                if self.is_person_in_elevator(state, p_id, e_id):
                    self.ee_success[p_id] += 1
            elif a_type == "EXIT":
                p_id = self.last_action[1]
                e_id = self.last_action[2]
                self.ee_attempts[p_id] += 1
                if not self.is_person_in_elevator(state, p_id, e_id):
                    self.ee_success[p_id] += 1
                    e_floor_before = self.get_elevator_floor(self.last_state, e_id)
                    if e_floor_before == self.game.get_person_goal(p_id):
                        reward = self.game.get_last_gained_reward()
                        if self.last_state[2] == 1:
                            reward -= self.game.get_goal_reward()
                        self.reward_sum[p_id] += reward
                        self.deliveries[p_id] += 1
                        self.total_deliveries += 1
                        
        self.last_state = state
        remaining_pids = [p for p, loc in state[1]]
        
        best_action = None
        best_val = -float('inf')
        
        for p in self.all_persons:
            is_rem = p in remaining_pids
            start_state = state if is_rem else self.game.get_initial_state()
            
            path, expected_steps = self.a_star(start_state, p)
            if expected_steps == float('inf'):
                continue
                
            ucb = self.get_ucb_reward(p)
            total_steps = expected_steps if is_rem else expected_steps + 1
            val = ucb / total_steps
            
            if val > best_val:
                best_val = val
                best_action = path[0] if (is_rem and path) else "RESET"
                
        sum_ucb_rem = sum(self.get_ucb_reward(p) for p in remaining_pids) + self.game.get_goal_reward()
        sum_steps_rem = sum(self.a_star(state, p)[1] for p in remaining_pids)
        if 0 < sum_steps_rem < float('inf'):
            val_all = sum_ucb_rem / sum_steps_rem
            if val_all > best_val:
                best_val = val_all
                best_rem_val = -float('inf')
                best_action = None
                for p in remaining_pids:
                    path, steps = self.a_star(state, p)
                    if steps < float('inf'):
                        val = self.get_ucb_reward(p) / steps
                        if val > best_rem_val:
                            best_rem_val = val
                            best_action = path[0] if path else "RESET"
                            
        if best_action is None:
            best_action = "RESET"
            
        if best_action == "RESET":
            action_str = "RESET"
            self.last_action = ("RESET",)
        else:
            action_str = f"{best_action[0]}{{{best_action[1]},{best_action[2]}}}"
            self.last_action = best_action
            
        return action_str
