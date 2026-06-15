import ext_elev
import heapq

id = ["000000000"] # Will be updated if needed

class Controller:
    """Dual-strategy pre-planned controller.
    Evaluates 'deploy everyone' vs 'farm highest reward person' and statically follows the optimal path.
    """
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.reachable = self.game.get_reachable()
        self.capacities = self.game.get_capacities()
        self.horizon = self.game.get_max_steps()
        self.goalReward = self.game.get_goal_reward()
        
        initial_state = self.game.get_initial_state()
        _, persons_t, _ = initial_state
        self.p_ids = [pid for pid, _ in persons_t]
        
        self.e_prob = {eid: self.game.get_elevator_action_prob(eid) for eid in self.reachable}
        self.p_prob = {pid: self.game.get_person_action_prob(pid) for pid in self.p_ids}
        self.p_weight = {pid: self.game.get_person_weight(pid) for pid in self.p_ids}
        self.p_goal = {pid: self.game.get_person_goal(pid) for pid in self.p_ids}
        self.p_reward = {pid: self.game.get_person_reward(pid) for pid in self.p_ids}
        self.p_avg_reward = {pid: sum(self.p_reward[pid]) / len(self.p_reward[pid]) for pid in self.p_ids}

        self.intersections = set()
        e_ids_list = sorted(list(self.reachable.keys()))
        for i, e1_id in enumerate(e_ids_list):
            e1_idx = [e[0] for e in initial_state[0]].index(e1_id)
            e1_set = set(self.reachable[e1_id]) | {initial_state[0][e1_idx][1]}
            for e2_id in e_ids_list[i+1:]:
                e2_idx = [e[0] for e in initial_state[0]].index(e2_id)
                e2_set = set(self.reachable[e2_id]) | {initial_state[0][e2_idx][1]}
                inter = e1_set & e2_set
                self.intersections.update(inter)

        self._precompute_all_min_transfers()
        self._precompute_best_elevator_transfers()
        
        rate1, plan1, states1 = self.eval_strategy_1()
        rate2, plan2, states2 = self.eval_strategy_2()
        
        print(f"Strategy 1 Rate: {rate1}, Plan: {plan1 is not None}")
        print(f"Strategy 2 Rate: {rate2}, Plan: {plan2 is not None}")
        
        if plan1 and rate1 > rate2:
            self.plan = plan1
            self.expected_states = states1
        elif plan2:
            self.plan = plan2
            self.expected_states = states2
        else:
            # Fallback
            self.plan = ["RESET"]
            self.expected_states = [initial_state]
            
        self.plan_idx = 0
        self.using_strategy_1 = (self.plan == plan1)
        self.plan2 = plan2
        self.states2 = states2

    def choose_next_action(self, state):
        initial_state = self.game.get_initial_state()
        
        # Hard-reset detection (e.g. engine testing a new seed)
        if state == initial_state:
            if self.plan_idx < len(self.plan) and self.expected_states[self.plan_idx] != initial_state:
                if self.plan_idx != 0:
                    self.plan_idx = 0
                    
        # Bailout logic
        if self.using_strategy_1 and self.plan_idx < len(self.plan):
            remaining_steps = self.game.get_max_steps() - self.game.get_current_steps()
            remaining_expected_cost = sum(1.0 / self.get_action_prob(self.plan[i]) for i in range(self.plan_idx, len(self.plan)))
            if remaining_expected_cost > remaining_steps:
                self.using_strategy_1 = False
                rate2, plan2, states2 = self.eval_strategy_2(state)
                if plan2:
                    self.plan = plan2
                    self.expected_states = states2
                    self.plan_idx = 0

        # Advance if previous action succeeded
        if self.plan_idx < len(self.plan):
            if state == self.expected_states[self.plan_idx]:
                self.plan_idx += 1
                
        # Loop the plan if finished
        if self.plan_idx >= len(self.plan):
            self.plan_idx = 0
            
        action = self.plan[self.plan_idx]
        
        # Bulletproof fallback against desyncs
        if action != "RESET":
            legal = self.get_legal_actions(state)
            if action not in legal:
                self.using_strategy_1 = False
                rate2, plan2, states2 = self.eval_strategy_2(state)
                if plan2:
                    self.plan = plan2
                    self.expected_states = states2
                    self.plan_idx = 0
                    action = self.plan[0]
                else:
                    action = "RESET"
            
        return action

    def astar_search(self, initial_state, is_goal_fn, get_actions_fn, heuristic_fn):
        counter = 0
        pq = []
        h_init = heuristic_fn(initial_state)
        heapq.heappush(pq, (h_init, counter, 0, initial_state, [], []))
        closed = set()
        
        while pq:
            f, _, g, state, path, state_path = heapq.heappop(pq)
            
            if is_goal_fn(state):
                print(f"A* Nodes expanded: {len(closed)}, generated: {counter}")
                return path, state_path
                
            if state in closed: continue
            closed.add(state)
            
            actions = get_actions_fn(state)
            for action in actions:
                succ = self._get_successor_state(state, action)
                if succ in closed: continue
                
                cost = 1.0 / self.get_action_prob(action)
                new_g = g + cost
                h = heuristic_fn(succ)
                new_f = new_g + h
                
                counter += 1
                heapq.heappush(pq, (new_f, counter, new_g, succ, path + [action], state_path + [succ]))
                
        return None, None

    def eval_strategy_1(self):
        init_state = self.game.get_initial_state()
        is_goal = lambda st: len(st[1]) == 0
        
        # We need a smaller action space for A* to run reasonably fast.
        # But we only run it once, so we can use a slightly optimized get_legal_actions.
        path, state_path = self.astar_search(init_state, is_goal, self.get_legal_actions, self.h_start)
        if not path:
            return -1, None, None
            
        expected_cost = sum(1.0 / self.get_action_prob(action) for action in path)
        expected_reward = self.goalReward + sum(self.p_avg_reward.values())
        rate = expected_reward / expected_cost
        
        return rate, path, state_path

    def eval_strategy_2(self, init_state=None):
        best_rate = -1
        best_path = None
        best_state_path = None
        best_p = None
        
        if init_state is None:
            init_state = self.game.get_initial_state()
            
        persons_t = init_state[1]
        available_p_ids = [p[0] for p in persons_t]
        
        for p_id in available_p_ids:
            is_goal = lambda st, pid=p_id: not any(p[0] == pid for p in st[1])
            get_actions = lambda st, pid=p_id: self.get_single_person_actions(st, pid)
            heuristic = lambda st, pid=p_id: self.h_single(st, pid)
            
            path, state_path = self.astar_search(init_state, is_goal, get_actions, heuristic)
            if not path: continue
                
            expected_cost = sum(1.0 / self.get_action_prob(action) for action in path) + 1.0 # +1 for RESET
            expected_reward = self.p_avg_reward[p_id]
            rate = expected_reward / expected_cost
            
            if rate > best_rate:
                best_rate = rate
                best_path = path + ["RESET"]
                best_state_path = state_path + [self.game.get_initial_state()]
                best_p = p_id
                
        return best_rate, best_path, best_state_path

    def get_legal_actions(self, state):
        elevators_t, persons_t, _ = state
        
        # 1. MANDATORY EXIT PRUNING:
        for p_id, p_loc in persons_t:
            if isinstance(p_loc, tuple) and p_loc[0] == 'in':
                e_id = p_loc[1]
                e_idx = None
                curr_f = None
                for i, (eid, f, w) in enumerate(elevators_t):
                    if eid == e_id:
                        e_idx = i
                        curr_f = f
                        break
                if curr_f == self.p_goal[p_id]:
                    return [f"EXIT{{{p_id},{e_id}}}"]

        actions = []
        e_ids_list = [eid for eid, _, _ in elevators_t]

        # 2. SELECTIVE MOVE:
        for j, (e_id, curr_f, _) in enumerate(elevators_t):
            reach = self.reachable[e_id]
            targets = set()

            # Pickup POIs
            for p_id, p_loc in persons_t:
                if isinstance(p_loc, tuple) and p_loc[0] == 'floor':
                    f_p = p_loc[1]
                    if f_p in reach:
                        targets.add(f_p)
            
            # Destination and Transfer POIs
            for p_id, p_loc in persons_t:
                if isinstance(p_loc, tuple) and p_loc[0] == 'in' and p_loc[1] == e_id:
                    goal = self.p_goal[p_id]
                    if goal in reach:
                        targets.add(goal)
                    else:
                        p_weight = self.p_weight[p_id]
                        matrix = self.weight_to_matrix[p_weight]
                        curr_t = matrix[curr_f][goal]
                        for f in self.intersections & set(reach):
                            if matrix[f][goal] < curr_t:
                                targets.add(f)
                        
            for t_f in targets:
                if t_f != curr_f:
                    actions.append(f"MOVE{{{e_id},{t_f}}}")

        # 3. ENTER and EXIT actions
        for p_id, p_loc in persons_t:
            p_weight = self.p_weight[p_id]
            if isinstance(p_loc, tuple) and p_loc[0] == 'floor':
                f_p = p_loc[1]
                goal = self.p_goal[p_id]
                matrix = self.weight_to_matrix[p_weight]
                curr_t = matrix[f_p][goal]
                
                can_direct_with_cap = any(
                    f == f_p and goal in self.reachable[eid] and w + p_weight <= self.capacities[eid]
                    for (eid, f, w) in elevators_t
                )
                
                for j, (e_id, e_f, e_w) in enumerate(elevators_t):
                    if e_f == f_p:
                        if e_w + p_weight > self.capacities[e_id]:
                            continue
                            
                        reach = self.reachable[e_id]
                        is_direct = goal in reach
                        
                        if can_direct_with_cap and not is_direct:
                            continue
                            
                        can_help = is_direct or any(matrix[f][goal] < curr_t for f in self.intersections & set(reach))
                        if can_help:
                            actions.append(f"ENTER{{{p_id},{e_id}}}")
            else:
                e_id = p_loc[1]
                curr_f = None
                for (eid, f, w) in elevators_t:
                    if eid == e_id:
                        curr_f = f
                        break
                if curr_f in self.intersections:
                    actions.append(f"EXIT{{{p_id},{e_id}}}")

        return actions

    def get_single_person_actions(self, state, target_p):
        elevators_t, persons_t, _ = state
        actions = []
        for (eid, cur_f, _) in elevators_t:
            for f in self.reachable[eid]:
                if f != cur_f: actions.append(f"MOVE{{{eid},{f}}}")
        for (pid, loc) in persons_t:
            if pid != target_p: continue
            if not (isinstance(loc, tuple) and loc[0] == 'floor'): continue
            f_p = loc[1]
            w_p = self.p_weight[pid]
            for (eid, cur_f, cur_w) in elevators_t:
                if cur_f == f_p and cur_w + w_p <= self.capacities[eid]:
                    actions.append(f"ENTER{{{pid},{eid}}}")
        for (pid, loc) in persons_t:
            if pid != target_p: continue
            if not (isinstance(loc, tuple) and loc[0] == 'in'): continue
            actions.append(f"EXIT{{{pid},{loc[1]}}}")
        return actions

    def get_action_prob(self, action):
        if action == "RESET": return 1.0
        elif action.startswith("MOVE"):
            e_id = int(action[5:-1].split(",")[0])
            return self.e_prob[e_id]
        elif action.startswith("ENTER"):
            p_id = int(action[6:-1].split(",")[0])
            return self.p_prob[p_id]
        elif action.startswith("EXIT"):
            p_id = int(action[5:-1].split(",")[0])
            return self.p_prob[p_id]
        return 1.0

    def _get_successor_state(self, state, action):
        if action == "RESET": return self.game.get_initial_state()
        elevators_t, persons_t, total_remaining = state
        elevators = list(elevators_t)
        persons = list(persons_t)
        
        if action.startswith("MOVE{"):
            e_id, target_f = map(int, action[5:-1].split(","))
            for i, (eid, floor, w) in enumerate(elevators):
                if eid == e_id:
                    elevators[i] = (eid, target_f, w)
                    break
        elif action.startswith("ENTER{"):
            p_id, e_id = map(int, action[6:-1].split(","))
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
            p_id, e_id = map(int, action[5:-1].split(","))
            p_weight = self.p_weight[p_id]
            e_floor = None
            for i, (eid, floor, w) in enumerate(elevators):
                if eid == e_id:
                    elevators[i] = (eid, floor, w - p_weight)
                    e_floor = floor
                    break
            
            if e_floor == self.p_goal[p_id]:
                persons = [p for p in persons if p[0] != p_id]
                new_remaining = total_remaining - 1
                return (tuple(sorted(elevators)), tuple(sorted(persons)), new_remaining)
            else:
                for i, (pid, loc) in enumerate(persons):
                    if pid == p_id:
                        persons[i] = (pid, ('floor', e_floor))
                        break
        return (tuple(sorted(elevators)), tuple(sorted(persons)), total_remaining)

    def h_single(self, state, target_p):
        elevators_t, persons_t, _ = state
        elev_positions = {eid: floor for (eid, floor, _) in elevators_t}
        for p_id, p_loc in persons_t:
            if p_id != target_p: continue
            goal = self.p_goal[p_id]
            p_weight = self.p_weight[p_id]
            matrix = self.weight_to_matrix[p_weight]
            unshareable = 0
            max_moves = 0
            if isinstance(p_loc, tuple) and p_loc[0] == 'floor':
                p_floor = p_loc[1]
                transfers = matrix[p_floor][goal]
                if transfers == float('inf'): return float('inf')
                unshareable += 2 * (transfers + 1)
                elevator_here = any(elev_positions.get(e_id) == p_floor and p_floor in self.reachable[e_id] and self.capacities[e_id] >= p_weight for e_id in self.reachable)
                max_moves = 1 if elevator_here else 2
            else:
                e_id = p_loc[1]
                reach = self.reachable[e_id]
                if goal in reach:
                    unshareable += 1
                    max_moves = 1 if elev_positions.get(e_id) != goal else 0
                else:
                    best_remaining = self.best_transfer_from_e[e_id][p_weight][goal]
                    if best_remaining == float('inf'): return float('inf')
                    unshareable += 1 + 2 * (best_remaining + 1)
                    max_moves = 1
            return unshareable + max_moves
        return 0

    def h_start(self, state):
        elevators_t, persons_t, _ = state
        elev_positions = {eid: floor for (eid, floor, _) in elevators_t}
        unshareable = 0
        max_moves = 0
        for p_id, p_loc in persons_t:
            goal = self.p_goal[p_id]
            p_weight = self.p_weight[p_id]
            matrix = self.weight_to_matrix[p_weight]
            if isinstance(p_loc, tuple) and p_loc[0] == 'floor':
                p_floor = p_loc[1]
                transfers = matrix[p_floor][goal]
                if transfers == float('inf'): return float('inf')
                unshareable += 2 * (transfers + 1)
                elevator_here = any(elev_positions.get(e_id) == p_floor and p_floor in self.reachable[e_id] and self.capacities[e_id] >= p_weight for e_id in self.reachable)
                max_moves = max(max_moves, 1 if elevator_here else 2)
            else:
                e_id = p_loc[1]
                reach = self.reachable[e_id]
                if goal in reach:
                    unshareable += 1
                    if elev_positions.get(e_id) != goal: max_moves = max(max_moves, 1)
                else:
                    best_remaining = self.best_transfer_from_e[e_id][p_weight][goal]
                    if best_remaining == float('inf'): return float('inf')
                    unshareable += 1 + 2 * (best_remaining + 1)
                    max_moves = max(max_moves, 1)
        return unshareable + max_moves

    def _precompute_all_min_transfers(self):
        unique_weights = set(self.p_weight.values())
        self.weight_to_matrix = {}
        for w in unique_weights:
            self.weight_to_matrix[w] = self._compute_matrix_for_weight(w)

    def _compute_matrix_for_weight(self, weight):
        height = max(max(reach) for reach in self.reachable.values())
        floors = list(range(height + 1))
        matrix = {i: {j: float('inf') for j in floors} for i in floors}
        for i in floors: matrix[i][i] = 0
        capable_elevators = [e_id for e_id in self.reachable if self.capacities[e_id] >= weight]
        initial_elevators = self.game.get_initial_state()[0]
        start_floors = {eid: floor for eid, floor, _ in initial_elevators}
        for e_id in capable_elevators:
            reach = list(self.reachable[e_id])
            start = start_floors.get(e_id, reach[0] if reach else 0)
            for f1 in reach:
                for f2 in reach: matrix[f1][f2] = 0
            if start not in self.reachable[e_id]:
                for f in reach: matrix[start][f] = 0
        for k in floors:
            for i in floors:
                if matrix[i][k] == float('inf'): continue
                for j in floors:
                    if matrix[k][j] == float('inf'): continue
                    new_t = matrix[i][k] + matrix[k][j] + 1
                    if new_t < matrix[i][j]: matrix[i][j] = new_t
        return matrix

    def _precompute_best_elevator_transfers(self):
        self.best_transfer_from_e = {}
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
                    if not reach:
                        self.best_transfer_from_e[e_id][w][goal] = float('inf')
                    else:
                        self.best_transfer_from_e[e_id][w][goal] = min(matrix[f][goal] for f in reach)
