"""
AI Disclosure:
This implementation and documentation were developed with the assistance of Gemini CLI.
"""

import ex1_check
import search as search
import utils as utils

# Update with your ID
id = ["212412258"]


class State:
    """
    Represents a state in the Elevators problem.
    
    To ensure hashability and efficiency, positions are stored as immutable tuples.
    Performance Optimizations:
    - Persons who have reached their goal are physically removed from the state 
      to shrink the search space.
    - personsPosition is stored as a sorted tuple of (p_id, location) to maintain 
      a consistent hash regardless of the order people are processed.
    """
    def __init__(self, elevatorsPosition, personsPosition, elevatorWeights):
        self.elevatorsPosition = tuple(elevatorsPosition)
        # Sort by person ID to ensure identical physical states have the same hash
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
        
        # 1. Initialize specifications (reachability, weight limits, intersections)
        self._initialize_specs(initial)
        
        # 2. Pre-calculate Weight-Aware Minimum Transfers Matrices.
        #    Calculating a separate matrix for each weight class ensures that 
        #    the heuristic and pruning logic respect physical weight limits.
        self._precompute_all_min_transfers()
        
        # 3. Pre-calculate best transfers from each elevator to each goal.
        #    This moves expensive min() calculations out of the h_astar hot loop.
        self._precompute_best_elevator_transfers()
        
        # 4. Create the initial state
        initial_elevators = [initial["Elevators"][e_id][0] for e_id in self.e_ids]
        initial_persons = []
        for p_id in self.p_ids:
            start_f = initial["Persons"][p_id][0]
            # State Dumping: Don't track people who are already at their goal
            if start_f != self.person_specs[p_id]["goal"]:
                initial_persons.append((p_id, start_f))
                
        initial_weights = [0] * len(self.e_ids)
        
        search.Problem.__init__(self, State(initial_elevators, initial_persons, initial_weights))

    def _initialize_specs(self, initial):
        """Parses the problem description into efficient lookup structures."""
        self.elevator_specs = {}
        for e_id in self.e_ids:
            curr_f, reach, weight = initial["Elevators"][e_id]
            self.elevator_specs[e_id] = {
                "reachable": frozenset(reach),
                "max_weight": weight
            }

        self.person_specs = {}
        for p_id in self.p_ids:
            start_f, w, goal_f = initial["Persons"][p_id]
            self.person_specs[p_id] = {
                "weight": w,
                "goal": goal_f
            }

        # Pre-identify intersection floors to facilitate transfers
        self.intersections = set()
        for i, e1_id in enumerate(self.e_ids):
            for e2_id in self.e_ids[i+1:]:
                inter = self.elevator_specs[e1_id]["reachable"] & self.elevator_specs[e2_id]["reachable"]
                self.intersections.update(inter)

    def _precompute_all_min_transfers(self):
        """Computes a unique min_transfers matrix for every unique passenger weight."""
        unique_weights = set(p["weight"] for p in self.person_specs.values())
        self.weight_to_matrix = {}
        for w in unique_weights:
            self.weight_to_matrix[w] = self._compute_matrix_for_weight(w)

    def _compute_matrix_for_weight(self, weight):
        """
        Computes the min transfers matrix using only elevators capable of 
        carrying the specified weight. Uses Floyd-Warshall with early stopping.
        """
        floors = list(range(self.height + 1))
        matrix = {i: {j: float('inf') for j in floors} for i in floors}
        
        for i in floors:
            matrix[i][i] = 0
            
        # Filter for elevators that can physically handle this weight
        capable_elevators = [e_id for e_id in self.e_ids if self.elevator_specs[e_id]["max_weight"] >= weight]
        
        for e_id in capable_elevators:
            reach = list(self.elevator_specs[e_id]["reachable"])
            for f1 in reach:
                for f2 in reach:
                    matrix[f1][f2] = 0
        
        while True:
            changed = False
            for k in floors:
                # Optimization: skip intermediate floors that are unreachable
                if all(matrix[i][k] == float('inf') for i in floors): continue
                for i in floors:
                    if matrix[i][k] == float('inf'): continue
                    for j in floors:
                        new_t = matrix[i][k] + matrix[k][j] + 1
                        if new_t < matrix[i][j]:
                            matrix[i][j] = new_t
                            changed = True
            if not changed:
                break
        return matrix

    def _precompute_best_elevator_transfers(self):
        """Pre-calculates the optimal transfer points for the heuristic function."""
        self.best_transfer_from_e = {} # Key: (e_id, weight, goal)
        all_goals = set(self.person_specs[p]["goal"] for p in self.p_ids)
        unique_weights = set(p["weight"] for p in self.person_specs.values())
        
        for e_id in self.e_ids:
            self.best_transfer_from_e[e_id] = {}
            reach = self.elevator_specs[e_id]["reachable"]
            for w in unique_weights:
                if self.elevator_specs[e_id]["max_weight"] < w: continue
                
                self.best_transfer_from_e[e_id][w] = {}
                matrix = self.weight_to_matrix[w]
                for goal in all_goals:
                    # Minimum transfers remaining from any floor this elevator can reach
                    self.best_transfer_from_e[e_id][w][goal] = min(matrix[f][goal] for f in reach)

    def successor(self, state):
        p_pos = state.personsPosition
        e_pos = state.elevatorsPosition
        e_weights = state.elevatorWeights

        # 1. MANDATORY EXIT PRUNING:
        # If a person can exit at their goal, they must. Return as the only successor.
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

        # 2. SELECTIVE MOVE:
        # Limit moves to floors with waiting people, passenger goals, or optimal intersections.
        for j, e_id in enumerate(self.e_ids):
            curr_f = e_pos[j]
            reach = self.elevator_specs[e_id]["reachable"]
            targets = set()

            # Pickup POIs
            for p_id, p_loc in p_pos:
                if isinstance(p_loc, int) and p_loc in reach:
                    targets.add(p_loc)
            
            # Destination and Transfer POIs
            for p_id, p_loc in p_pos:
                if p_loc == f"e{e_id}":
                    goal = self.person_specs[p_id]["goal"]
                    if goal in reach:
                        targets.add(goal)
                    else:
                        # Only move to intersections that bring this specific passenger closer to their goal
                        p_weight = self.person_specs[p_id]["weight"]
                        matrix = self.weight_to_matrix[p_weight]
                        curr_t = matrix[curr_f][goal]
                        for f in self.intersections & reach:
                            if matrix[f][goal] < curr_t:
                                targets.add(f)
                        
            for t_f in targets:
                if t_f != curr_f:
                    new_e_pos = list(e_pos)
                    new_e_pos[j] = t_f
                    successors.append((f"MOVE{{{e_id},{t_f}}}", State(tuple(new_e_pos), p_pos, e_weights)))

        # 3. ENTER and EXIT actions
        for p_id, p_loc in p_pos:
            p_weight = self.person_specs[p_id]["weight"]
            if isinstance(p_loc, int):
                # Selective ENTER with Dynamic Capacity-Awareness:
                goal = self.person_specs[p_id]["goal"]
                matrix = self.weight_to_matrix[p_weight]
                curr_t = matrix[p_loc][goal]
                
                # Check if a direct elevator is at the floor and has room
                can_direct_with_cap = any(e_pos[k] == p_loc and goal in self.elevator_specs[eid]["reachable"] 
                                          and e_weights[k] + p_weight <= self.elevator_specs[eid]["max_weight"]
                                          for k, eid in enumerate(self.e_ids))
                
                for j, e_id in enumerate(self.e_ids):
                    if e_pos[j] == p_loc:
                        if e_weights[j] + p_weight > self.elevator_specs[e_id]["max_weight"]:
                            continue
                            
                        reach = self.elevator_specs[e_id]["reachable"]
                        is_direct = goal in reach
                        
                        # Optimization: Skip transfer-elevators if a direct one is present with capacity
                        if can_direct_with_cap and not is_direct:
                            continue
                            
                        # Only enter if the elevator provides a path with the min transfers for this person
                        can_help = is_direct or any(matrix[f][goal] < curr_t for f in self.intersections & reach)
                        
                        if can_help:
                            new_p_pos = tuple((pid, (f"e{e_id}" if pid == p_id else loc)) for pid, loc in p_pos)
                            new_weights = list(e_weights)
                            new_weights[j] += p_weight
                            successors.append((f"ENTER{{{p_id},{e_id}}}", State(e_pos, new_p_pos, tuple(new_weights))))
            else:
                # Transfer EXIT: Exit at intersection points to switch elevators
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
        # Goal is reached when the personsPosition tuple is empty due to physical dumping
        return len(state.personsPosition) == 0

    def h_astar(self, node):
        """
        Admissible and highly informative heuristic based on weight-aware transfers.
        Calculates exact ENTER/EXIT requirements + shared move penalties.
        """
        state = node.state
        unshareable = 0
        max_moves = 0
        
        for p_id, p_loc in state.personsPosition:
            goal = self.person_specs[p_id]["goal"]
            p_weight = self.person_specs[p_id]["weight"]
            matrix = self.weight_to_matrix[p_weight]
            
            if isinstance(p_loc, int):
                # Calculate required ENTER/EXIT pairs based on min transfers
                transfers = matrix[p_loc][goal]
                if transfers == float('inf'): return float('inf') # Impossible goal
                unshareable += 2 * (transfers + 1)
                
                # Check for elevator availability at the pickup floor
                elevator_here = False
                for j, e_id in enumerate(self.e_ids):
                    if state.elevatorsPosition[j] == p_loc and p_loc in self.elevator_specs[e_id]["reachable"] \
                       and self.elevator_specs[e_id]["max_weight"] >= p_weight:
                        elevator_here = True
                        break
                # Penalize based on distance to the first elevator
                max_moves = max(max_moves, 1 if elevator_here else 2)
            else:
                # In elevator: 1 (EXIT) + 2 * remaining transfers
                e_id = int(p_loc[1:])
                e_idx = self.e_ids.index(e_id)
                reach = self.elevator_specs[e_id]["reachable"]
                
                if goal in reach:
                    unshareable += 1
                    if state.elevatorsPosition[e_idx] != goal:
                        max_moves = max(max_moves, 1)
                else:
                    best_remaining = self.best_transfer_from_e[e_id][p_weight][goal]
                    if best_remaining == float('inf'): return float('inf')
                    unshareable += 1 + 2 * (best_remaining + 1)
                    max_moves = max(max_moves, 1)
                    
        return unshareable + max_moves


def create_elevators_problem(game):
    return ElevatorsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
