import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from match import Match


class Schedule:
    def __init__(self, n_teams: int):
        # n_teams: Integer representing the number of teams in the tournament (must be even)
        # match: Dictionary with the matches of the tournament as key and the week assigned as value

        super().__init__()

        self.n_teams: int = n_teams
        self.matches: dict = dict()

        # Make a circle of the n_teams - 1 teams and assign the week for each match
        for team_1 in range(self.n_teams - 2):
            self.matches[Match(team_1, team_1 + 1)] = team_1 + 1

        self.matches[Match(0, n_teams - 2)] = self.n_teams - 1

        # Assign the week for the other matches by taking the parallel match
        for team_1 in range(self.n_teams - 1):
            for team_2 in range(team_1 + 2, self.n_teams - 1):
                if team_1 != 0 or team_2 != self.n_teams - 2:
                    if team_2 - team_1 < n_teams / 2:
                        team_1_neighbour = team_1 - 1 if team_1 > 0 else self.n_teams - 2
                        team_2_neighbour = team_2 + 1 if team_2 < self.n_teams - 2 else 0
                    else:
                        team_1_neighbour = team_1 + 1 if team_1 < self.n_teams - 2 else 0
                        team_2_neighbour = team_2 - 1 if team_2 > 0 else self.n_teams - 2

                    if team_1_neighbour < team_2_neighbour:
                        self.matches[Match(team_1, team_2)] = self.matches[Match(team_1_neighbour, team_2_neighbour)]
                    else:
                        self.matches[Match(team_1, team_2)] = self.matches[Match(team_2_neighbour, team_1_neighbour)]

        # Assign weeks for the last team matches
        for team in range(self.n_teams - 1):
            team_weeks =\
                [self.matches[match] for match in self.matches.keys() if team in match]
            self.matches[Match(team, self.n_teams - 1)] = \
                [week for week in range(1, self.n_teams) if week not in team_weeks][0]

        self.matches = dict(sorted(self.matches.items(), key=lambda item: item[1]))
        print(self.matches)

    def matches_to_tuples(self):
        # Convert the matches dictionary to a list of tuples

        return [match.to_tuple() for match in self.matches.keys()]

    def show_as_graph(self):
        # Show the schedule as a graph

        schedule: nx.Graph = nx.Graph()
        schedule.add_edges_from(self.matches_to_tuples())

        pos = nx.spring_layout(schedule)

        nx.draw_networkx(schedule, pos, node_shape="s", node_color="none",
                         bbox=dict(facecolor="skyblue", edgecolor='none', boxstyle='round,pad=0.2'))

        nx.draw_networkx_edge_labels(
            schedule, pos,
            edge_labels={match.to_tuple(): self.matches[match] for match in self.matches.keys()},
            font_color='blue'
        )

        plt.axis('off')
        plt.show()

    def show_as_ndarray(self):
        # Show the schedule as a numpy ndarray

        schedule: np.ndarray = np.array(list(self.matches.keys()))
        schedule = schedule.reshape(self.n_teams - 1, (int(self.n_teams / 2)))

        print(schedule, schedule.shape)


if __name__ == '__main__':
    schedule = Schedule(6)
    schedule.show_as_graph()
    schedule.show_as_ndarray()
