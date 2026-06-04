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
        self.p_goal = {pid: self.game.get_person_goal(pid) for pid in person_ids}
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

        # Greedy selection: pick actions that minimize the successor's h_start heuristic
        best_actions = []
        min_h = float('inf')
        
        for action in actions:
            # We don't want to RESET unless it's the only option or optimal
            if action == "RESET" and len(actions) > 1:
                continue
            
            successor = self._get_successor_state(state, action)
            h_val = self.h_start(successor)
            
            if h_val < min_h:
                min_h = h_val
                best_actions = [action]
            elif h_val == min_h:
                best_actions.append(action)
                
        if not best_actions:
            best_actions = ["RESET"]
            
        return str(np.random.choice(best_actions))

    def _get_successor_state(self, state, action):
        """Returns the successor state assuming the action succeeds."""
        if action == "RESET":
            return self.game.get_initial_state()
            
        elevators_t, persons_t, total_remaining = state
        
        elevators = list(elevators_t)
        persons = list(persons_t)
        
        if action.startswith("MOVE{"):
            parts = action[5:-1].split(",")
            e_id = int(parts[0])
            target_f = int(parts[1])
            for i, (eid, floor, w) in enumerate(elevators):
                if eid == e_id:
                    elevators[i] = (eid, target_f, w)
                    break
                    
        elif action.startswith("ENTER{"):
            parts = action[6:-1].split(",")
            p_id = int(parts[0])
            e_id = int(parts[1])
            p_weight = self.p_weight[p_id]
            for i, (eid, floor, w) in enumerate(elevators):
                if eid == e_id:
                    elevators[i] = (eid, floor, w + p_weight)
                    break
            for i, (pid, loc) in enumerate(persons):
                if pid == p_id:
                    persons[i] = (pid, ('in', e_id))
                    break
                    
        elif action.startswith("EXIT{"):
            parts = action[5:-1].split(",")
            p_id = int(parts[0])
            e_id = int(parts[1])
            p_weight = self.p_weight[p_id]
            e_floor = None
            for i, (eid, floor, w) in enumerate(elevators):
                if eid == e_id:
                    elevators[i] = (eid, floor, w - p_weight)
                    e_floor = floor
                    break
            
            # Check if delivered
            goal = self.p_goal[p_id]
            if e_floor == goal:
                # Delivered: remove from persons
                persons = [p for p in persons if p[0] != p_id]
                new_remaining = total_remaining - 1
                if new_remaining == 0:
                    return self.game.get_initial_state()
                else:
                    return (tuple(sorted(elevators)), tuple(sorted(persons)), new_remaining)
            else:
                for i, (pid, loc) in enumerate(persons):
                    if pid == p_id:
                        persons[i] = (pid, ('floor', e_floor))
                        break
                        
        return (tuple(sorted(elevators)), tuple(sorted(persons)), total_remaining)
    def h_start(self, state):
        """
        Admissible and highly informative heuristic based on weight-aware transfers.
        Calculates exact ENTER/EXIT requirements + shared move penalties.
        """
        if not hasattr(self, 'weight_to_matrix'):
            self._precompute_all_min_transfers()
            self._precompute_best_elevator_transfers()

        elevators_t, persons_t, _ = state

        # Mapping from elevator ID to its current floor
        # elevators_t elements are (eid, floor, cur_weight)
        elev_positions = {eid: floor for (eid, floor, _) in elevators_t}

        unshareable = 0
        max_moves = 0
        
        for p_id, p_loc in persons_t:
            goal = self.p_goal[p_id]
            p_weight = self.p_weight[p_id]
            matrix = self.weight_to_matrix[p_weight]
            
            # Check if person is on a floor
            if isinstance(p_loc, tuple) and p_loc[0] == 'floor':
                p_floor = p_loc[1]
                # Calculate required ENTER/EXIT pairs based on min transfers
                transfers = matrix[p_floor][goal]
                if transfers == float('inf'): return float('inf') # Impossible goal
                unshareable += 2 * (transfers + 1)
                
                # Check for elevator availability at the pickup floor
                elevator_here = False
                for e_id in self.reachable:
                    if elev_positions.get(e_id) == p_floor and p_floor in self.reachable[e_id] \
                       and self.capacities[e_id] >= p_weight:
                        elevator_here = True
                        break
                # Penalize based on distance to the first elevator
                max_moves = max(max_moves, 1 if elevator_here else 2)
            else:
                # In elevator: loc is ('in', e_id)
                e_id = p_loc[1]
                reach = self.reachable[e_id]
                
                if goal in reach:
                    unshareable += 1
                    if elev_positions.get(e_id) != goal:
                        max_moves = max(max_moves, 1)
                else:
                    best_remaining = self.best_transfer_from_e[e_id][p_weight][goal]
                    if best_remaining == float('inf'): return float('inf')
                    unshareable += 1 + 2 * (best_remaining + 1)
                    max_moves = max(max_moves, 1)
                    
        return unshareable + max_moves

    def _precompute_all_min_transfers(self):
        """Computes a unique min_transfers matrix for every unique passenger weight."""
        unique_weights = set(self.p_weight.values())
        self.weight_to_matrix = {}
        for w in unique_weights:
            self.weight_to_matrix[w] = self._compute_matrix_for_weight(w)

    def _compute_matrix_for_weight(self, weight):
        """
        Computes a directed min transfers matrix.
        Accounts for the fact that an elevator can pick up at its starting floor
        even if it's not in its bi-directional reachable set.
        """
        height = max(max(reach) for reach in self.reachable.values())
        floors = list(range(height + 1))
        matrix = {i: {j: float('inf') for j in floors} for i in floors}
        
        for i in floors:
            matrix[i][i] = 0
            
        capable_elevators = [e_id for e_id in self.reachable if self.capacities[e_id] >= weight]
        
        # Get start floors from initial state
        initial_elevators = self.game.get_initial_state()[0]
        start_floors = {eid: floor for eid, floor, _ in initial_elevators}
        
        for e_id in capable_elevators:
            reach = list(self.reachable[e_id])
            start = start_floors[e_id]
            
            # Bi-directional movement between all reachable floors
            for f1 in reach:
                for f2 in reach:
                    matrix[f1][f2] = 0
            
            # One-way movement from starting floor to reachable floors (if not already reachable)
            if start not in self.reachable[e_id]:
                for f in reach:
                    matrix[start][f] = 0
        
        # Floyd-Warshall (Directed)
        for k in floors:
            for i in floors:
                if matrix[i][k] == float('inf'): continue
                for j in floors:
                    if matrix[k][j] == float('inf'): continue
                    new_t = matrix[i][k] + matrix[k][j] + 1
                    if new_t < matrix[i][j]:
                        matrix[i][j] = new_t
        return matrix

    def _precompute_best_elevator_transfers(self):
        """Pre-calculates the optimal transfer points for the heuristic function."""
        self.best_transfer_from_e = {} # Key: (e_id, weight, goal)
        all_goals = set(self.p_goal.values())
        unique_weights = set(self.p_weight.values())
        
        for e_id in self.reachable:
            self.best_transfer_from_e[e_id] = {}
            reach = self.reachable[e_id]
            for w in unique_weights:
                if self.capacities[e_id] < w: continue
                
                self.best_transfer_from_e[e_id][w] = {}
                matrix = self.weight_to_matrix[w]
                for goal in all_goals:
                    # Minimum transfers remaining from any floor this elevator can reach
                    self.best_transfer_from_e[e_id][w][goal] = min(matrix[f][goal] for f in reach)
