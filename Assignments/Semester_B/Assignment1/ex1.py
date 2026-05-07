import ex1_check
import search as search
import utils as utils

id = ["No numbers - I'm special!"]


class State:
    def __init__(self, elevatorsPosition, personsPosition, elevatorWeights):
        self.elevatorsPosition = tuple(elevatorsPosition)
        self.personsPosition = tuple(personsPosition)
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
        self.person_specs = {}
        self.goal_elevators = {}
        self.intersections = {}

        initial_elevators = []
        for i, e_id in enumerate(self.e_ids):
            curr_f, reach, weight = initial["Elevators"][e_id]
            self.elevator_specs[e_id] = {"reachable": frozenset(reach), "max_weight": weight}
            initial_elevators.append(curr_f)

        initial_persons = []
        for p_id in self.p_ids:
            start_f, w, goal_f = initial["Persons"][p_id]
            self.person_specs[p_id] = {"weight": w, "goal": goal_f}
            self.goal_elevators[p_id] = [e for e in self.e_ids if goal_f in self.elevator_specs[e]["reachable"]]
            initial_persons.append(start_f)

        for e1 in self.e_ids:
            for e2 in self.e_ids:
                if e1 != e2:
                    self.intersections[(e1, e2)] = self.elevator_specs[e1]["reachable"] & self.elevator_specs[e2]["reachable"]

        self.needs_transfer = {}
        for p_id in self.p_ids:
            start_f = initial["Persons"][p_id][0]
            goal_f = self.person_specs[p_id]["goal"]
            can_direct = False
            for e_id in self.e_ids:
                reach = self.elevator_specs[e_id]["reachable"]
                if start_f in reach and goal_f in reach:
                    can_direct = True
                    break
            self.needs_transfer[p_id] = not can_direct

        initial_weights = [0] * len(self.e_ids)
        search.Problem.__init__(self, State(initial_elevators, initial_persons, initial_weights))

    def successor(self, state):
        p_pos = state.personsPosition
        e_pos = state.elevatorsPosition
        e_weights = state.elevatorWeights

        for i, p_id in enumerate(self.p_ids):
            p_loc = p_pos[i]
            if isinstance(p_loc, str):
                e_id = int(p_loc[1:])
                e_idx = self.e_ids.index(e_id)
                if e_pos[e_idx] == self.person_specs[p_id]["goal"]:
                    new_p_pos = list(p_pos)
                    new_p_pos[i] = e_pos[e_idx]
                    new_weights = list(e_weights)
                    new_weights[e_idx] -= self.person_specs[p_id]["weight"]
                    return [(f"EXIT{{{p_id},{e_id}}}", State(e_pos, new_p_pos, new_weights))]

        successors = []
        waiting_floors = set()
        for i, p_id in enumerate(self.p_ids):
            if isinstance(p_pos[i], int) and p_pos[i] != self.person_specs[p_id]["goal"]:
                waiting_floors.add(p_pos[i])

        for i, e_id in enumerate(self.e_ids):
            curr_f = e_pos[i]
            passengers = [k for k, loc in enumerate(p_pos) if loc == f"e{e_id}"]
            
            targets = set()
            if not passengers:
                targets = waiting_floors & self.elevator_specs[e_id]["reachable"]
            else:
                for k in passengers:
                    p_goal = self.person_specs[self.p_ids[k]]["goal"]
                    if p_goal in self.elevator_specs[e_id]["reachable"]:
                        targets.add(p_goal)
                    else:
                        for other_e in self.goal_elevators[self.p_ids[k]]:
                            targets.update(self.intersections.get((e_id, other_e), set()))
                targets.update(waiting_floors & self.elevator_specs[e_id]["reachable"])

            for t_f in targets:
                if t_f != curr_f:
                    new_e_pos = list(e_pos)
                    new_e_pos[i] = t_f
                    successors.append((f"MOVE{{{e_id},{t_f}}}", State(new_e_pos, p_pos, e_weights)))

        for i, p_id in enumerate(self.p_ids):
            p_loc = p_pos[i]
            if p_loc == self.person_specs[p_id]["goal"]:
                continue
            
            p_weight = self.person_specs[p_id]["weight"]
            if isinstance(p_loc, int):
                for j, e_id in enumerate(self.e_ids):
                    if e_pos[j] == p_loc:
                        if e_weights[j] + p_weight <= self.elevator_specs[e_id]["max_weight"]:
                            new_p_pos = list(p_pos)
                            new_p_pos[i] = f"e{e_id}"
                            new_weights = list(e_weights)
                            new_weights[j] += p_weight
                            successors.append((f"ENTER{{{p_id},{e_id}}}", State(e_pos, new_p_pos, new_weights)))
            else:
                e_id = int(p_loc[1:])
                e_idx = self.e_ids.index(e_id)
                curr_f = e_pos[e_idx]
                p_goal = self.person_specs[p_id]["goal"]
                if p_goal not in self.elevator_specs[e_id]["reachable"]:
                    for other_e in self.goal_elevators[p_id]:
                        if curr_f in self.elevator_specs[other_e]["reachable"]:
                            new_p_pos = list(p_pos)
                            new_p_pos[i] = curr_f
                            new_weights = list(e_weights)
                            new_weights[e_idx] -= p_weight
                            successors.append((f"EXIT{{{p_id},{e_id}}}", State(e_pos, new_p_pos, new_weights)))
        return successors

    def goal_test(self, state):
        for i, p_id in enumerate(self.p_ids):
            if state.personsPosition[i] != self.person_specs[p_id]["goal"]:
                return False
        return True

    def h_astar(self, node):
        state = node.state
        unshareable = 0
        max_moves = 0
        for i, p_id in enumerate(self.p_ids):
            p_loc = state.personsPosition[i]
            goal = self.person_specs[p_id]["goal"]
            if p_loc == goal:
                continue
            
            needs_tr = self.needs_transfer[p_id]
            if isinstance(p_loc, int):
                unshareable += 4 if needs_tr else 2
                
                can_take_p = False
                for j, e_id in enumerate(self.e_ids):
                    if state.elevatorsPosition[j] == p_loc:
                        reach = self.elevator_specs[e_id]["reachable"]
                        if p_loc in reach:
                            if goal in reach or (needs_tr and any(f in reach for f in self.goal_elevators[p_id])):
                                can_take_p = True
                                break
                max_moves = max(max_moves, 1 if can_take_p else 2)
            else:
                unshareable += 3 if needs_tr else 1
                e_id = int(p_loc[1:])
                e_idx = self.e_ids.index(e_id)
                if state.elevatorsPosition[e_idx] != goal:
                    max_moves = max(max_moves, 1)
        return unshareable + max_moves


def create_elevators_problem(game):
    print("<<create_elevators_problem")
    """ Create an elevators problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return ElevatorsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
