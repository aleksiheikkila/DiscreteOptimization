#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from tqdm import tqdm


Item = namedtuple("Item", ['index', 'value', 'weight'])


# Maximum cells (k * j) that is solved using DP approach
MAX_CELLS_FOR_DP = 50_000_000


def process_input(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    return item_count, capacity, items


def format_output(value: float, taken: list[int], opt: int = 0):
    # prepare the solution in the specified output format
    output_data = str(int(value)) + ' ' + str(int(opt)) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def solve_it_greedy_density(input_data):
    """Start adding items beginning from the most value dense ones. 
    Add to the knapsack if it fits, oterhwise go to the next item
    """
    item_count, capacity, items = process_input(input_data)

    value = weight = 0
    taken = [0] * len(items)

    items_in_density_order = items.copy()
    items_in_density_order = sorted(items_in_density_order, key=lambda x: -float(x[1]) / x[2])  # all weights positive non-zero

    # print(items_in_density_order)

    for item in items_in_density_order:
        # add if space
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    print("total value obtained:", value)

    # prepare the solution in the specified output format
    output_data = format_output(value, taken, opt=0)

    return output_data


def solve_it_dp(input_data):
    item_count, capacity, items = process_input(input_data)
    print("capa:", capacity)

    # print(items)

    # value = weight = 0


    ######################################
    # Algo
    # dp[k, j] = max value if the capacity was k and one can use items 0..., j
    
    # capacity on rows, item
    dp = np.zeros(shape=(capacity + 1, item_count + 1))

    for j in tqdm(range(1, item_count + 1)):
        for k in range(1, capacity + 1):
            # would j'th item fit?
            item = items[j-1]
            value_j = item.value
            weight_j = item.weight

            if weight_j <= k:
                # can fit
                dp[k, j] = max(
                    dp[k, j-1], 
                    value_j + dp[k-weight_j, j-1]
                )
            else:
                # does not fit
                dp[k, j] = dp[k, j-1]

    print(dp)

    ######################################
    value = dp[-1, -1]
    print("total value obtained:", value)

    # Then construct the taken (which items were selected)
    # start for the dp[-1, -1]
    # if cell value is zero, we know no more items were selected
    # if not:
    # check the cell to the left. If it has same value as current cell, 
    # we did not select current item (no value added)
    # if it is different, we added the current item
    # then reduce the weight of that item (go up wj) and go one left

    taken = [0] * len(items)

    k = capacity 
    j = item_count
    v = dp[k, j]

    while v > 0 and k > 0 and j > 0:
        # print((k, j))
        # check left
        if dp[k, j] == dp[k, j-1]:
            # jth item was not selected
            j -= 1  # go one left
        else:
            # jth item selected
            taken[j-1] = 1
            k -= items[j-1].weight
            j -= 1

    print("taken:", taken)

    # prepare the solution in the specified output format
    # opt = 1 means that the solution is optimal!
    output_data = format_output(value, taken, opt=1)

    return output_data


@dataclass
class Node:
    item: Item  # item to be considered for selection
    taken: list[int]
    current_value: float
    capacity_left: float
    optimistic_estimate: float = None
    level: int = None

    def get_opt_est(self, items_in_desc_value_density_order: list[Item]) -> float:
        self.optimistic_estimate = get_optimistic_estimate(
            capacity_left=self.capacity_left,
            current_value=self.current_value,
            taken=self.taken,
            items_in_desc_value_density_order=items_in_desc_value_density_order
        )
        return self.optimistic_estimate


def get_optimistic_estimate(
    capacity_left: int,
    current_value: float,
    taken: list[int],
    items_in_desc_value_density_order: list[Item],
    current_item_level: int
) -> float:
    # THIS VERSION IS FOR DFS IN DESC DENSITY ORDER!
    # select items starting from highest value density
    # until the knapsack is filled (can take a fraction)

    # if capacity_left <= 0:
    #     return current_value  # cannot add value
    
    value_estimate = current_value

    for item in items_in_desc_value_density_order[current_item_level:]:
        if capacity_left <= 0:
            break

        # if item.index < current_item_level:
        #     # already past this one, skip
        #     continue

        assert taken[item.index] == 0
        # if taken[item.index] == 1:
        #     # already taken, go to next item
        #     continue

        # else select it, either fully or partially
        take = min(1, capacity_left / item.weight)

        value_estimate += (take * item.value)
        capacity_left -= (take * item.weight)

    return value_estimate


def solve_it_bb_with_relaxation_dfs(input_data):
    # Branch and bound with linear relaxation

    # Sort it into desc value density order first, and do DFS in that
    # kind of a graph

    # Bounding:
    # linear relaxation: can select fraction of objects
    # calculate the optimistic value (relaxation) using that
    # in a greedy way: start from the most value-dense items
    # Take a fraction of the last item to fill the capacity

    # Other search alternatives:
    # Best First, 
    # Limited Discrepancy Search

    item_count, capacity, items = process_input(input_data)

    # Sort to desc value density
    items_in_desc_value_density_order = items.copy()
    items_in_desc_value_density_order = sorted(
        items_in_desc_value_density_order, 
        key=lambda x: -float(x[1]) / x[2]
    )  # all weights positive non-zero

    best_value = 0
    best_taken = [0] * item_count

    # DFS
    # Stack s
    s = [Node(
            item=items_in_desc_value_density_order[0],
            taken=[0] * item_count,
            current_value=0,
            capacity_left=capacity,
            level=0,
        )  
    ]

    while s:
        node = s.pop()
        item = node.item

        capacity_left = node.capacity_left
        current_value = node.current_value
        current_level = node.level
        taken = node.taken

        # if at lowest level (final decision)
        if current_level == item_count - 1:
            # add it to the knapsack if possible
            if item.weight <= capacity_left:
                assert taken[item.index] == 0
                taken[item.index] = 1
                current_value += item.value

            # check optimality so far
            if current_value > best_value:
                best_value = current_value
                best_taken = taken.copy()
                # print("NEW BEST:", best_value)
                # print("SELECTIONS:", best_taken)

        else:
            # not at lowest level

            # Calculate optimistic estimate (upper bound)
            # If it is less than the best value seen so far, do not continue
            opt_est = get_optimistic_estimate(
                capacity_left,
                current_value,
                taken=taken,
                items_in_desc_value_density_order=items_in_desc_value_density_order,
                current_item_level=current_level
            )

            if opt_est <= best_value:
                # print(f"Pruning as optimistic estimate: {opt_est} <= best so far: {best_value}")
                continue

            # Check if we can add the item or not
            # If yes, add both branches to the stack (i.e. select, do not select)
            # If no, add just the right branch (i.e. do not select)

            new_level = current_level + 1 

            # add right (do not select) always
            s.append(Node(
                item=items_in_desc_value_density_order[new_level],
                taken=taken.copy(),
                current_value=current_value,
                capacity_left=capacity_left,
                level=new_level,
                )
            )

            # can the current item be selected? If yes, add that branch to the stack as well
            if item.weight <= capacity_left:
                # Add left, i.e. select this
                assert taken[item.index] == 0
                taken_new = taken.copy()
                taken_new[item.index] = 1
                new_value = current_value + item.value
                new_capacity_left = capacity_left - item.weight
                s.append(Node(
                    item=items_in_desc_value_density_order[new_level],
                    taken=taken_new,
                    current_value=new_value,
                    capacity_left=new_capacity_left,
                    level=new_level,
                    )
                )

    print(f"FINAL BEST VALUE: {best_value}")
    print(f"with TAKEN: {best_taken}")

    # generate the output
    output_data = format_output(best_value, best_taken, opt=1)

    return output_data
 
    
def solve_it(input_data):
    # TODO: Reformat (process_input is done twice)

    item_count, capacity, items = process_input(input_data)

    # If small enough problem, use DP
    # Else branch and bound

    if item_count * capacity <= MAX_CELLS_FOR_DP:
        return solve_it_dp(input_data)
    else:
        # replace this with relaxation branch and bound
        # return solve_it_greedy_density(input_data)
        # return solve_it_greedy_density(input_data)
        return solve_it_bb_with_relaxation_dfs(input_data)
    


def solve_it_almost_orig(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    item_count, capacity, items = process_input(input_data)

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    
    print("total value obtained:", value)

    # prepare the solution in the specified output format
    output_data = format_output(value, taken, opt=0)

    return output_data



if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
