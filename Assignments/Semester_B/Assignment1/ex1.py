import ex1_check
import search as search
import utils as utils

id = ["No numbers - I'm special!"]


class State:
    def __init__(self, elevatorsPosition, personsPosition):
        self.elevatorsPosition = tuple(elevatorsPosition)
        self.personsPosition = tuple(personsPosition)

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

        initial_elevators = []
        for e_id in self.e_ids:
            curr_f, reach, weight = initial["Elevators"][e_id]
            self.elevator_specs[e_id] = {"reachable": frozenset(reach), "max_weight": weight}
            initial_elevators.append(curr_f)

        initial_persons = []
        for p_id in self.p_ids:
            start_f, w, goal_f = initial["Persons"][p_id]
            self.person_specs[p_id] = {"weight": w, "goal": goal_f}
            initial_persons.append(start_f)

        search.Problem.__init__(self, State(initial_elevators, initial_persons))

    def successor(self, state):
        successors = []
        for i, p_id in enumerate(self.p_ids):
            p_loc = state.personsPosition[i]
            if not isinstance(p_loc, int):
                e_id = int(p_loc[1:])
                e_idx = self.e_ids.index(e_id)
                if state.elevatorsPosition[e_idx] == self.person_specs[p_id]["goal"]:
                    new_p_pos = list(state.personsPosition)
                    new_p_pos[i] = state.elevatorsPosition[e_idx]
                    return [(f"EXIT{{{p_id},{e_id}}}", State(state.elevatorsPosition, new_p_pos))]

        for i, e_id in enumerate(self.e_ids):
            curr_f = state.elevatorsPosition[i]
            for target_f in self.elevator_specs[e_id]["reachable"]:
                if target_f != curr_f:
                    new_e_pos = list(state.elevatorsPosition)
                    new_e_pos[i] = target_f
                    successors.append((f"MOVE{{{e_id},{target_f}}}", State(new_e_pos, state.personsPosition)))

        for i, p_id in enumerate(self.p_ids):
            p_loc = state.personsPosition[i]
            if p_loc == self.person_specs[p_id]["goal"]:
                continue
            
            p_weight = self.person_specs[p_id]["weight"]
            if isinstance(p_loc, int):
                for j, e_id in enumerate(self.e_ids):
                    if state.elevatorsPosition[j] == p_loc:
                        e_max = self.elevator_specs[e_id]["max_weight"]
                        curr_w = sum(self.person_specs[self.p_ids[k]]["weight"] 
                                     for k, loc in enumerate(state.personsPosition) 
                                     if loc == f"e{e_id}")
                        if curr_w + p_weight <= e_max:
                            new_p_pos = list(state.personsPosition)
                            new_p_pos[i] = f"e{e_id}"
                            successors.append((f"ENTER{{{p_id},{e_id}}}", State(state.elevatorsPosition, new_p_pos)))
            else:
                e_id = int(p_loc[1:])
                e_idx = self.e_ids.index(e_id)
                curr_f = state.elevatorsPosition[e_idx]
                new_p_pos = list(state.personsPosition)
                new_p_pos[i] = curr_f
                successors.append((f"EXIT{{{p_id},{e_id}}}", State(state.elevatorsPosition, new_p_pos)))
        return successors

    def goal_test(self, state):
        for i, p_id in enumerate(self.p_ids):
            if state.personsPosition[i] != self.person_specs[p_id]["goal"]:
                return False
        return True

    def h_astar(self, node):
        state = node.state
        cost = 0
        for i, p_id in enumerate(self.p_ids):
            p_loc = state.personsPosition[i]
            goal = self.person_specs[p_id]["goal"]
            if p_loc != goal:
                if isinstance(p_loc, int):
                    cost += 3
                else:
                    cost += 2
        return cost


def create_elevators_problem(game):
    print("<<create_elevators_problem")
    """ Create an elevators problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return ElevatorsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
