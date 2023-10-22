import sys
import numpy as np

from match import Match
from schedule import Schedule


def get_swappable_matches(schedule: Schedule, tabou_list: list[tuple[Match, Match]]) -> set[tuple[Match, Match]]:
    conflicting_matches: dict[int] = dict()

    for period in schedule.matches.T:
        for match_1 in period:
            # A match is in conflict if one of the two teams is already playing two times in the period
            team_1_conflicts_number: int = 0
            team_2_conflicts_number: int = 0

            match_is_in_conflict: bool = False

            match_1_index: int = int(np.where(period == match_1)[0])
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
                    if (match_2, match_1) not in swappable_matches:
                        swappable_matches.add((match_1, match_2))

    return swappable_matches


def tabou_search(init_schedule: Schedule, tabu_max_length: int, max_iteration: int) -> Schedule:
    tabou_list: list[tuple[Match, Match]] = []  # List of tabou matches
    schedule: Schedule = init_schedule.deepcopy()  # Copy of the initial schedule
    iteration: int = 0

    # Flag to check if the schedule verifies all constraints
    schedule_is_valid: bool = False

    # Performe the tabou search until the schedule is valid or the max number of iterations is reached
    while (iteration < max_iteration) or not schedule_is_valid:
        # Matches that can be swapped in a week (at least one of the two matches is in conflict)
        swappable_matches: set[tuple[Match, Match]] = get_swappable_matches(schedule, tabou_list)

        iteration += 1

    # for week in schedule:
    #     for period in week:
    #         if period.conflict(period):
    #             return False

    return schedule


def main(argv):
    n_teams = int(argv[0])  # Number of teams

    if n_teams % 2 != 0:
        raise ValueError('Number of teams must be even and greater than 0!')

    # Instantiate a schedule array of dimensions (n_teams - 1, n_teams / 2)
    # with the weeks as rows and the periods as columns
    schedule: Schedule = Schedule(n_teams)
    print(schedule.matches, np.shape(schedule.matches))

    tabou_search(schedule, n_teams, 10000)


if __name__ == '__main__':
    main(sys.argv[1:])
