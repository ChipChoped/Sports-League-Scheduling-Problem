import sys
from typing import Tuple, Set

import numpy as np

from match import Match
from schedule import Schedule


def find_matches_in_conflict(schedule: Schedule) -> set[tuple[Match, Match]]:
    conflicting_matches: dict[int] = dict()

    for period in schedule.matches.T:
        for match_1 in period:
            # A match is in conflict if one of the two teams is already playing two times in the period
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
    for match_1, match_2 in tabu_list:
        if (match_1, match_2) in matches_in_conflict:
            matches_in_conflict.remove((match_1, match_2))
        elif (match_2, match_1) in matches_in_conflict:
            matches_in_conflict.remove((match_2, match_1))

    return matches_in_conflict


def best_neighbour_schedule(schedule: Schedule, swappable_matches: set[tuple[Match, Match]],
                            tabu_list: list[tuple[Match, Match]]) \
        -> tuple[tuple[Match, Match], Schedule, set[tuple[Match, Match]], set[tuple[Match, Match]]]:
    updated_swappable_matches: set[tuple[Match, Match]] = get_swappable_matches(swappable_matches, tabu_list)

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

    # Search for the swap leading to the least conflicting neighbourhood
    updated_swappable_matches.remove(best_swap)

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

        if len(neighbour_matches_in_conflict) < len(best_neighbour_matches_in_conflict):
            best_swap = swap
            best_neighbour = neighbour
            best_neighbour_matches_in_conflict = neighbour_matches_in_conflict
            best_neighbour_swappable_matches = neighbour_swappable_matches

    return best_swap, best_neighbour, best_neighbour_matches_in_conflict, best_neighbour_swappable_matches


def tabu_search(init_schedule: Schedule, tabu_list_length: int, max_iterations: int) -> Schedule:
    tabu_list: list[tuple[Match, Match]] = list()  # List of tabu matches
    schedule: Schedule = init_schedule.deepcopy()  # Copy of the initial schedule
    iteration: int = 0
    swap: tuple[Match, Match] = (Match(0, 0), Match(0, 0))

    # Flag to check if the schedule verifies all constraints
    schedule_is_valid: bool = False

    # Initialize the tabu_list queue with placeholders
    for _ in range(tabu_list_length - 1):
        tabu_list.append((Match(0, 0), Match(0, 0)))

    # Matches that can be swapped in a week (at least one of the two matches is in conflict)
    matches_in_conflict: set[tuple[Match, Match]] = find_matches_in_conflict(schedule)
    swappable_matches: set[tuple[Match, Match]] = get_swappable_matches(matches_in_conflict, tabu_list)

    print("i =", iteration)
    print("t =", len(tabu_list), tabu_list)
    print("c =", len(swappable_matches))
    print("s =", swap)
    print(schedule.matches)
    print("")

    # Perform the tabu search until the schedule is valid and the max number of iterations is reached
    while not schedule_is_valid and iteration < max_iterations:

        swap, schedule, matches_in_conflict, swappable_matches = \
            best_neighbour_schedule(schedule, swappable_matches, tabu_list)

        if len(matches_in_conflict) == 0:
            schedule_is_valid = True
            print(matches_in_conflict)
        elif len(tabu_list) > 0:
            tabu_list.pop(0)

        tabu_list.append(swap)

        print("i =", iteration)
        print("t =", len(tabu_list), tabu_list)
        print("c =", len(matches_in_conflict), matches_in_conflict)
        print("s =", len(swappable_matches), swappable_matches)
        print("m =", swap)
        print(schedule.matches)
        print("")

        iteration += 1

    return schedule


def main(argv):
    n_teams = int(argv[0])  # Number of teams
    max_iterations = int(argv[1])  # Maximum number of iterations

    if n_teams % 2 != 0:
        raise ValueError('Number of teams must be even and greater than 0!')

    # Instantiate a schedule array of dimensions (n_teams - 1, n_teams / 2)
    # with the weeks as rows and the periods as columns
    schedule: Schedule = Schedule(n_teams)

    valid_schedule = tabu_search(schedule, int((n_teams / 2) ** 2), max_iterations)

    print(schedule.matches, np.shape(schedule.matches))
    print("")
    print(valid_schedule.matches, np.shape(valid_schedule.matches))


if __name__ == '__main__':
    main(sys.argv[1:])
