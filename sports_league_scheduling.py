import sys

import numpy as np

from match import Match
from schedule import Schedule


def get_swappable_matches(schedule: Schedule, tabu_list: list[tuple[Match, Match]]) -> set[tuple[Match, Match]]:
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

    swappable_matches: set[tuple[Match, Match]] = set()

    for week in schedule.matches:
        week_index: int = int(np.where(schedule.matches == week)[0][0])

        if week_index in conflicting_matches:
            for match_1 in conflicting_matches[week_index]:
                match_1_index, = np.where(week == match_1)[0]
                week_copy: np.ndarray = np.delete(week.copy(), match_1_index)

                for match_2 in week_copy:
                    if (match_1, match_2) not in tabu_list and (match_2, match_1) not in tabu_list\
                            and (match_2, match_1) not in swappable_matches:
                        swappable_matches.add((match_1, match_2))

    return swappable_matches


def best_neighbour_schedule(schedule: Schedule, swappable_matches: set[tuple[Match, Match]],
                            tabu_list: list[tuple[Match, Match]]) -> tuple[tuple[Match, Match], Schedule, set[tuple[Match, Match]]]:
    best_swap = next(iter(swappable_matches))

    match_1_i, match_1_j = np.where(schedule.matches == best_swap[0])
    match_2_i, match_2_j = np.where(schedule.matches == best_swap[1])

    best_neighbour: Schedule = schedule.deepcopy()

    best_neighbour.matches[match_1_i[0]][match_1_j[0]] = best_swap[1]
    best_neighbour.matches[match_2_i[0]][match_2_j[0]] = best_swap[0]

    best_neighbour_swappable_matches: set[tuple[Match, Match]] = get_swappable_matches(best_neighbour, tabu_list)

    # Search for the swap leading to the least conflicting neighbourhood
    for swap in swappable_matches:
        match_1_i, match_1_j = np.where(schedule.matches == swap[0])
        match_2_i, match_2_j = np.where(schedule.matches == swap[1])

        neighbour: Schedule = schedule.deepcopy()

        neighbour.matches[match_1_i[0]][match_1_j[0]] = swap[1]
        neighbour.matches[match_2_i[0]][match_2_j[0]] = swap[0]

        neighbour_swappable_matches: set[tuple[Match, Match]] = get_swappable_matches(best_neighbour, tabu_list)

        if len(neighbour_swappable_matches) < len(best_neighbour_swappable_matches):
            best_swap = swap
            best_neighbour = neighbour
            best_neighbour_swappable_matches = neighbour_swappable_matches

    return best_swap, best_neighbour, best_neighbour_swappable_matches


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
    swappable_matches: set[tuple[Match, Match]] = get_swappable_matches(schedule, tabu_list)

    # Perform the tabu search until the schedule is valid and the max number of iterations is reached
    while not schedule_is_valid and iteration < max_iterations:
        print("i =", iteration)
        print("t =", len(tabu_list), tabu_list)
        print("c =", len(swappable_matches))
        print("s =", swap)
        print("")

        swap, schedule, swappable_matches = best_neighbour_schedule(schedule, swappable_matches, tabu_list)

        if len(tabu_list) == 0:
            if len(swappable_matches) == 0:
                schedule_is_valid = True
        else:
            tabu_list.pop(0)

        tabu_list.append(swap)
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
    valid_schedule = tabu_search(schedule, n_teams * 2, max_iterations)

    print(schedule.matches, np.shape(schedule.matches))
    print(valid_schedule.matches, np.shape(valid_schedule.matches))


if __name__ == '__main__':
    main(sys.argv[1:])
