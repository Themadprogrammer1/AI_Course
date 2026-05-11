import ex1_check
import search as search
import utils as utils

# Update with your ID
id = ["No numbers - I'm special!"]


class State:
    def __init__(self, elevatorsPosition, personsPosition, elevatorWeights):
        self.elevatorsPosition = tuple(elevatorsPosition)
        # personsPosition is a tuple of (p_id, location)
        self.personsPosition = tuple(sorted(personsPosition))
        self.elevatorWeights = tuple(elevatorWeights)

    def __eq__(self, other):
        return isinstance(other, State) and \
               self.elevatorsPosition == other.elevatorsPosition and \
               self.personsPosition == other.personsPosition

    def __hash__(self):
        return hash((self.elevatorsPosition, self.personsPosition))


class ElevatorsProblem(search.Problem):
    def __init__(self, initial):
        self.height = initial["height"]
        self.e_ids = tuple(sorted(initial["Elevators"].keys()))
        self.p_ids = tuple(sorted(initial["Persons"].keys()))
        
        self.elevator_specs = {}
        for e_id in self.e_ids:
            curr_f, reach, weight = initial["Elevators"][e_id]
            self.elevator_specs[e_id] = {
                "reachable": frozenset(reach),
                "max_weight": weight
            }

        self.person_specs = {}
        required_pairs = []
        for p_id in self.p_ids:
            start_f, w, goal_f = initial["Persons"][p_id]
            self.person_specs[p_id] = {
                "weight": w,
                "goal": goal_f
            }
            if start_f != goal_f:
                required_pairs.append((start_f, goal_f))

        # 1. Initialize Min Transfers Matrix
        floors = list(range(self.height + 1))
        self.min_transfers = {i: {j: float('inf') for j in floors} for i in floors}
        
        for i in floors:
            self.min_transfers[i][i] = 0
            
        for e_id in self.e_ids:
            reach = list(self.elevator_specs[e_id]["reachable"])
            for f1 in reach:
                for f2 in reach:
                    self.min_transfers[f1][f2] = 0
        
        # 2. DP Loop (Floyd-Warshall Style with early stopping)
        while True:
            if all(self.min_transfers[s][g] < float('inf') for s, g in required_pairs):
                break
            
            changed = False
            for k in floors:
                for i in floors:
                    for j in floors:
                        new_t = self.min_transfers[i][k] + self.min_transfers[k][j] + 1
                        if new_t < self.min_transfers[i][j]:
                            self.min_transfers[i][j] = new_t
                            changed = True
            if not changed:
                break

        self.intersections = set()
        for i, e1_id in enumerate(self.e_ids):
            for e2_id in self.e_ids[i+1:]:
                inter = self.elevator_specs[e1_id]["reachable"] & self.elevator_specs[e2_id]["reachable"]
                self.intersections.update(inter)

        initial_elevators = [initial["Elevators"][e_id][0] for e_id in self.e_ids]
        
        initial_persons = []
        for p_id in self.p_ids:
            start_f = initial["Persons"][p_id][0]
            if start_f != self.person_specs[p_id]["goal"]:
                initial_persons.append((p_id, start_f))
                
        initial_weights = [0] * len(self.e_ids)
        
        search.Problem.__init__(self, State(initial_elevators, initial_persons, initial_weights))

    def successor(self, state):
        p_pos = state.personsPosition
        e_pos = state.elevatorsPosition
        e_weights = state.elevatorWeights

        # 1. Pruning: If a person is at their goal in an elevator, EXIT is the ONLY successor
        for p_id, p_loc in p_pos:
            if isinstance(p_loc, str):
                e_id = int(p_loc[1:])
                e_idx = self.e_ids.index(e_id)
                if e_pos[e_idx] == self.person_specs[p_id]["goal"]:
                    new_p_pos = tuple(p for p in p_pos if p[0] != p_id)
                    new_weights = list(e_weights)
                    new_weights[e_idx] -= self.person_specs[p_id]["weight"]
                    return [(f"EXIT{{{p_id},{e_id}}}", State(e_pos, new_p_pos, tuple(new_weights)))]

        successors = []

        # 2. MOVE: Only to points of interest
        for j, e_id in enumerate(self.e_ids):
            curr_f = e_pos[j]
            reach = self.elevator_specs[e_id]["reachable"]
            
            targets = set()
            for p_id, p_loc in p_pos:
                if isinstance(p_loc, int) and p_loc in reach:
                    targets.add(p_loc)
            # Drop-off and Transfer
            for p_id, p_loc in p_pos:
                if p_loc == f"e{e_id}":
                    goal = self.person_specs[p_id]["goal"]
                    if goal in reach:
                        targets.add(goal)
                    else:
                        # Only move to intersections that bring us closer to the goal
                        curr_t = self.min_transfers[curr_f][goal]
                        for f in self.intersections & reach:
                            if self.min_transfers[f][goal] < curr_t:
                                targets.add(f)
                        
            for t_f in targets:
                if t_f != curr_f:
                    new_e_pos = list(e_pos)
                    new_e_pos[j] = t_f
                    successors.append((f"MOVE{{{e_id},{t_f}}}", State(tuple(new_e_pos), p_pos, e_weights)))

        # 3. ENTER and EXIT at intersection
        for p_id, p_loc in p_pos:
            p_weight = self.person_specs[p_id]["weight"]
            if isinstance(p_loc, int):
                # ENTER
                goal = self.person_specs[p_id]["goal"]
                curr_t = self.min_transfers[p_loc][goal]
                for j, e_id in enumerate(self.e_ids):
                    if e_pos[j] == p_loc:
                        reach = self.elevator_specs[e_id]["reachable"]
                        can_enter = False
                        if goal in reach:
                            can_enter = True
                        else:
                            for f in self.intersections & reach:
                                if self.min_transfers[f][goal] < curr_t:
                                    can_enter = True
                                    break
                        if not can_enter:
                            continue
                            
                        if e_weights[j] + p_weight <= self.elevator_specs[e_id]["max_weight"]:
                            new_p_pos = tuple((pid, (f"e{e_id}" if pid == p_id else loc)) for pid, loc in p_pos)
                            new_weights = list(e_weights)
                            new_weights[j] += p_weight
                            successors.append((f"ENTER{{{p_id},{e_id}}}", State(e_pos, new_p_pos, tuple(new_weights))))
            else:
                # EXIT at intersection (goal exits handled by pruning above)
                e_id = int(p_loc[1:])
                e_idx = self.e_ids.index(e_id)
                curr_f = e_pos[e_idx]
                if curr_f in self.intersections:
                    new_p_pos = tuple((pid, (curr_f if pid == p_id else loc)) for pid, loc in p_pos)
                    new_weights = list(e_weights)
                    new_weights[e_idx] -= p_weight
                    successors.append((f"EXIT{{{p_id},{e_id}}}", State(e_pos, new_p_pos, tuple(new_weights))))

        return successors

    def goal_test(self, state):
        return len(state.personsPosition) == 0

    def h_astar(self, node):
        state = node.state
        unshareable = 0
        max_moves = 0
        
        for p_id, p_loc in state.personsPosition:
            goal = self.person_specs[p_id]["goal"]
            
            if isinstance(p_loc, int):
                transfers = self.min_transfers[p_loc][goal]
                unshareable += 2 * (transfers + 1)
                
                elevator_here = False
                for j, e_id in enumerate(self.e_ids):
                    if state.elevatorsPosition[j] == p_loc and p_loc in self.elevator_specs[e_id]["reachable"]:
                        elevator_here = True
                        break
                max_moves = max(max_moves, 1 if elevator_here else 2)
            else:
                e_id = int(p_loc[1:])
                e_idx = self.e_ids.index(e_id)
                reach = self.elevator_specs[e_id]["reachable"]
                
                if goal in reach:
                    unshareable += 1
                    if state.elevatorsPosition[e_idx] != goal:
                        max_moves = max(max_moves, 1)
                else:
                    best_remaining = min(self.min_transfers[f][goal] for f in reach)
                    unshareable += 1 + 2 * (best_remaining + 1)
                    max_moves = max(max_moves, 1)
                    
        return unshareable + max_moves


def create_elevators_problem(game):
    return ElevatorsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
