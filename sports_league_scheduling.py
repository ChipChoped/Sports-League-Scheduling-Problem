import sys
import time
import random

import numpy as np

from match import Match
from schedule import Schedule


def find_matches_in_conflict(schedule: Schedule) -> set[tuple[Match, Match]]:
    # Find the matches in conflict in a schedule

    conflicting_matches: dict[int] = dict()

    for period in schedule.matches.T:
        for match_1 in period:
            # Count the number of conflicts for each team in the match
            team_1_conflicts_number: int = 0
            team_2_conflicts_number: int = 0

            match_is_in_conflict: bool = False

            match_1_index: int = int(np.where(period == match_1)[0][0])
            period_copy: np.ndarray = np.delete(period.copy(), match_1_index)

            for match_2 in period_copy:
                conflict: tuple[bool, bool] = match_1.teams_conflict(match_2)

                if conflict[0]:
                    team_1_conflicts_number += 1

                if conflict[1]:
                    team_2_conflicts_number += 1

                # If a team is in conflict more than once, the match is in conflict in the schedule
                # and the loop can be stopped
                if team_1_conflicts_number > 1 or team_2_conflicts_number > 1:
                    match_is_in_conflict = True
                    break

            if match_is_in_conflict:
                if match_1_index not in conflicting_matches:
                    conflicting_matches[match_1_index] = set()

                conflicting_matches[match_1_index].add(match_1)

    matches_in_conflict: set[tuple[Match, Match]] = set()

    for week in schedule.matches:
        week_index: int = int(np.where(schedule.matches == week)[0][0])

        if week_index in conflicting_matches:
            for match_1 in conflicting_matches[week_index]:
                match_1_index, = np.where(week == match_1)[0]
                week_copy: np.ndarray = np.delete(week.copy(), match_1_index)

                for match_2 in week_copy:
                    matches_in_conflict.add((match_1, match_2))

    return matches_in_conflict


def get_swappable_matches(matches_in_conflict: set[tuple[Match, Match]], tabu_list: list[tuple[Match, Match]]) \
        -> set[tuple[Match, Match]]:
    # Remove from the set of tuples of matches in conflict the ones that are in the tabu list

    swappable_matches: set[tuple[Match, Match]] = matches_in_conflict.copy()

    for match_1, match_2 in tabu_list:
        if (match_1, match_2) in swappable_matches:
            swappable_matches.remove((match_1, match_2))
        elif (match_2, match_1) in swappable_matches:
            swappable_matches.remove((match_2, match_1))

    return swappable_matches


def best_neighbour_schedule(schedule: Schedule, matches_in_conflict: set[tuple[Match, Match]],
                            tabu_list: list[tuple[Match, Match]]) \
        -> tuple[tuple[Match, Match], Schedule, set[tuple[Match, Match]], set[tuple[Match, Match]]]:
    # Search for the swap leading to the least conflicting neighbourhood

    # Get the set of swappable matches for the current schedule
    updated_swappable_matches: set[tuple[Match, Match]] = get_swappable_matches(matches_in_conflict, tabu_list)

    # If no swappable matches are found, remove the oldest element from the tabu list and try again
    while len(updated_swappable_matches) == 0:
        tabu_list.pop(0)
        tabu_list.append((Match(0, 0), Match(0, 0)))

        updated_swappable_matches = get_swappable_matches(matches_in_conflict, tabu_list)

    # Choose the first swap from the set of swappable matches as the best swap temporarily
    best_swap = next(iter(updated_swappable_matches))

    match_1_i, match_1_j = np.where(schedule.matches == best_swap[0])
    match_2_i, match_2_j = np.where(schedule.matches == best_swap[1])

    best_neighbour: Schedule = schedule.deepcopy()

    best_neighbour.matches[match_1_i[0]][match_1_j[0]] = best_swap[1]
    best_neighbour.matches[match_2_i[0]][match_2_j[0]] = best_swap[0]

    best_neighbour_matches_in_conflict: set[tuple[Match, Match]] = \
        find_matches_in_conflict(best_neighbour)
    best_neighbour_swappable_matches: set[tuple[Match, Match]] = \
        get_swappable_matches(best_neighbour_matches_in_conflict, tabu_list)

    updated_swappable_matches.remove(best_swap)

    # Search for the best swap in the neighbourhood
    for swap in updated_swappable_matches:
        match_1_i, match_1_j = np.where(schedule.matches == swap[0])
        match_2_i, match_2_j = np.where(schedule.matches == swap[1])

        neighbour: Schedule = schedule.deepcopy()

        neighbour.matches[match_1_i[0]][match_1_j[0]] = swap[1]
        neighbour.matches[match_2_i[0]][match_2_j[0]] = swap[0]

        neighbour_matches_in_conflict: set[tuple[Match, Match]] = \
            find_matches_in_conflict(neighbour)
        neighbour_swappable_matches: set[tuple[Match, Match]] = \
            get_swappable_matches(neighbour_matches_in_conflict, tabu_list)

        # If the number of conflicts is lower, update the best swap
        if len(neighbour_matches_in_conflict) < len(best_neighbour_matches_in_conflict):
            best_swap = swap
            best_neighbour = neighbour
            best_neighbour_matches_in_conflict = neighbour_matches_in_conflict
            best_neighbour_swappable_matches = neighbour_swappable_matches

    return best_swap, best_neighbour, best_neighbour_matches_in_conflict, best_neighbour_swappable_matches


def random_swap(schedule: Schedule, matches_in_conflict: set[tuple[Match, Match]], tabu_list: list[tuple[Match, Match]]) \
        -> tuple[tuple[Match, Match], Schedule, set[tuple[Match, Match]], set[tuple[Match, Match]]]:
    updated_swappable_matches: set[tuple[Match, Match]] = \
        get_swappable_matches(matches_in_conflict, tabu_list)

    # If no swappable matches are found, remove the oldest element from the tabu list and try again
    while len(updated_swappable_matches) == 0:
        tabu_list.pop(0)
        tabu_list.append((Match(0, 0), Match(0, 0)))

        updated_swappable_matches = \
            get_swappable_matches(matches_in_conflict, tabu_list)

    # Choose a random swap from the set of swappable matches
    swap = random.choice(tuple(updated_swappable_matches))

    match_1_i, match_1_j = np.where(schedule.matches == swap[0])
    match_2_i, match_2_j = np.where(schedule.matches == swap[1])

    neighbour: Schedule = schedule.deepcopy()

    neighbour.matches[match_1_i[0]][match_1_j[0]] = swap[1]
    neighbour.matches[match_2_i[0]][match_2_j[0]] = swap[0]

    neighbour_matches_in_conflict: set[tuple[Match, Match]] = \
        find_matches_in_conflict(neighbour)
    neighbour_swappable_matches: set[tuple[Match, Match]] = \
        get_swappable_matches(neighbour_matches_in_conflict, tabu_list)

    return swap, neighbour, neighbour_matches_in_conflict, neighbour_swappable_matches


def evaluate_neighbour_schedule(schedule: Schedule) -> tuple[int, tuple[int, int] | None]:
    # Evaluate the number of conflicts in a schedule

    conflicts: int = 0
    t = None

    for period in schedule.matches.T:
        for team in range(schedule.n_teams):
            c: int = 0

            for match in period:
                if team in match:
                    c += 1

            # Check if the team is playing more than two times in the period
            if c > 2:
                conflicts += c - 2
                t = (period, team)

    return conflicts, t


def tabu_search(init_schedule: Schedule, tabu_list_length: int, max_iterations: int) -> tuple[Schedule, int, bool]:
    # Perform a tabu search on a schedule

    tabu_list: list[tuple[Match, Match]] = list()  # List of tabu matches
    schedule: Schedule = init_schedule.deepcopy()  # Copy of the initial schedule
    iteration: int = 0  # Current iteration
    swap: tuple[Match, Match] = (Match(0, 0), Match(0, 0))  # Last swap performed

    # Flag to check if the schedule verifies all constraints
    schedule_is_valid: bool = False
    best_schedule: Schedule = schedule.deepcopy()

    stagnation_counter: int = 0  # Counter to check if the algorithm is stuck in a local minimum

    # Initialize the tabu_list queue with placeholders
    for _ in range(tabu_list_length - 1):
        tabu_list.append((Match(0, 0), Match(0, 0)))

    # Matches that can be swapped in a week (at least one of the two matches is in conflict)
    matches_in_conflict: set[tuple[Match, Match]] = find_matches_in_conflict(schedule)
    swappable_matches: set[tuple[Match, Match]] = get_swappable_matches(matches_in_conflict, tabu_list)

    best_value: int = len(matches_in_conflict)

    # print("i =", iteration)
    # print("t =", len(tabu_list), tabu_list)
    # print("c =", len(swappable_matches))
    # print("s =", swap)
    # print(schedule.matches)
    # print("")

    # Perform the tabu search until the schedule is valid and the max number of iterations is reached
    while not schedule_is_valid and iteration < max_iterations:
        # Mean to avoid being stuck in a local minimum
        if stagnation_counter > tabu_list_length * 4:
            schedule = best_schedule.deepcopy()
            matches_in_conflict = find_matches_in_conflict(schedule)

            swap, schedule, matches_in_conflict, swappable_matches = \
                random_swap(schedule, matches_in_conflict, tabu_list)

            # swap, schedule, matches_in_conflict, swappable_matches, tabu_swappable_matches, schedule_evaluation = \
            #     best_neighbour_schedule(schedule, matches_in_conflict, swappable_matches, schedule_evaluation, tabu_list)

            # tabu_list = tabu_list_initialization(tabu_list_length)
            stagnation_counter = 0

        # Get a new schedule
        else:
            swap, schedule, matches_in_conflict, swappable_matches = \
                best_neighbour_schedule(schedule, matches_in_conflict, tabu_list)

        # Check if the schedule is valid
        if len(matches_in_conflict) == 0:
            schedule_is_valid = True
        # Remove the oldest element from the tabu list if not empty
        elif len(tabu_list) > 0:
            tabu_list.pop(0)

        # Save the best schedule if found and reset the stagnation counter
        if len(matches_in_conflict) < best_value:
            best_schedule = schedule.deepcopy()
            best_value = len(matches_in_conflict)
            stagnation_counter = 0
        # Increment the stagnation counter if no improvement is found
        else:
            stagnation_counter += 1

        # Add the new swap to the tabu list
        tabu_list.append(swap)

        # print("i =", iteration)
        # print("t =", len(tabu_list), tabu_list)
        # print("c =", len(matches_in_conflict))
        # print("s =", len(swappable_matches))
        # print("m =", swap)
        # print("e =", evaluate_neighbour_schedule(schedule))
        # # print(schedule.matches)
        # print("")

        iteration += 1

    return best_schedule, iteration, schedule_is_valid


def main(argv):
    n_teams = int(argv[0])  # Number of teams
    max_iterations = int(argv[1])  # Maximum number of iterations
    tabu_list_length = 1  # Length of the tabu list
    is_test: bool = False
    n_tests: int = 1
    n_success: int = 0
    total_time: time = 0
    total_iterations: int = 0

    if len(argv) > 2:
        tabu_list_length = int(argv[2])

    if len(argv) > 3:
        is_test = True
        n_tests = int(argv[3])  # Number of tests

    if n_teams % 2 != 0:
        raise ValueError('Number of teams must be even and greater than 0!')

    test_iterations: int = 0

    if is_test:
        print("Number of teams:", n_teams)
        print("Tabu list length:", tabu_list_length)
        print("")

    while test_iterations < n_tests:
        # Instantiate a schedule array of dimensions (n_teams - 1, n_teams / 2)
        # with the weeks as rows and the periods as columns

        start: time = time.time()
        schedule: Schedule = Schedule(n_teams)
        valid_schedule, iteration, schedule_is_valide = tabu_search(schedule, tabu_list_length, max_iterations)
        end: time = time.time()

        tuples = valid_schedule.matches_to_tuples()

        if is_test:
            total_time += end - start

        # print(schedule.matches, np.shape(schedule.matches))
        # print("")
        # print(valid_schedule.matches, np.shape(valid_schedule.matches))

        if is_test:
            if schedule_is_valide:
                n_success += 1
                total_iterations += iteration

            print("Test n°", test_iterations + 1)
            print("Number of successes:", n_success)
            print("Success rate:", n_success / (test_iterations + 1) * 100, "%")
            print("Total number of iterations:", total_iterations)

            if n_success > 0:
                print("Average number of iterations:", int(total_iterations / n_success))

            print("Average number of iterations_:", int(total_iterations / (test_iterations + 1)))
            print("Total time:", round(total_time, 3), "s")
            print("Average time:", round(total_time / (test_iterations + 1), 3), "s")
        else:
            tuples = valid_schedule.matches_to_tuples()
            # print(evaluate_neighbour_schedule(valid_schedule))

            print("Number of teams:", n_teams)
            print("Tabu list length:", tabu_list_length)
            print("Iterations:", iteration + 1, "/", max_iterations)
            print("Time:", round(end - start, 3), "s")

        if schedule_is_valide:
            print("A solution was found:")
        else:
            print("Number of conflicts to resolve:", evaluate_neighbour_schedule(valid_schedule))
            print("No solution was found under", max_iterations, "iterations", "\nThe best schedule found is:")

        for j in range(0, n_teams - 1):
            for i in range(0, int(n_teams / 2)):
                print(tuples[j * int(n_teams / 2) + i], end=' ')

            print("")

        print("")

        test_iterations += 1

    if is_test:
        print("Number of teams:", n_teams)
        print("Tabu list length:", tabu_list_length)
        print("Maximum number of iterations:", max_iterations)
        print("Number of tests:", n_tests)
        print("Number of successes:", n_success)
        print("Success rate:", n_success / n_tests * 100, "%")
        print("Total number of iterations:", total_iterations)
        print("Average number of iterations:", int(total_iterations / n_success))
        print("Average number of iterations_:", int(total_iterations / total_iterations))
        print("Total time:", round(total_time, 3), "s")
        print("Average time:", round(total_time / n_tests, 3), "s")
        print("")


if __name__ == '__main__':
    main(sys.argv[1:])
