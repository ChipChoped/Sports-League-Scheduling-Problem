import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from match import Match


class Schedule:
    def __init__(self, n_teams: int = None):
        # self.n_teams: Integer representing the number of teams in the tournament (must be even)
        # self.matches: NdArray of Match objects representing the matches of the tournament where
        #               the rows respect the week constraint

        self.n_teams: int
        self.matches: np.ndarray

        if n_teams is not None:
            self.n_teams = n_teams
            weeks_matches_init: dict = dict()

            # Make a circle of the n_teams - 1 teams and assign the week for each match
            for team_1 in range(self.n_teams - 2):
                weeks_matches_init[Match(team_1, team_1 + 1)] = team_1 + 1

            weeks_matches_init[Match(0, n_teams - 2)] = self.n_teams - 1

            # Assign the week for the other matches by taking the parallel match
            for team_1 in range(self.n_teams - 1):
                even_k: int = int(self.n_teams - 5 - (self.n_teams - 6) / 2) if self.n_teams > 6 else 1
                odd_k: int = 1

                is_even: bool = True

                for team_2 in range(team_1 + 2, self.n_teams - 1):
                    if team_1 != 0 or team_2 != self.n_teams - 2:
                        if is_even:
                            team_1_neighbour = team_1 - even_k if team_1 - even_k >= 0 \
                                else self.n_teams - 1 - even_k + team_1
                            team_2_neighbour = team_2 + even_k if team_2 + even_k <= self.n_teams - 2 \
                                else 0 + even_k - (self.n_teams - 1 - team_2)

                            even_k -= 1
                            is_even = False
                        else:
                            team_1_neighbour = team_1 + odd_k if team_1 + odd_k <= self.n_teams - 2 \
                                else 0 + odd_k - (self.n_teams - 1 - team_1)
                            team_2_neighbour = team_2 - odd_k if team_2 - odd_k > 0 \
                                else self.n_teams - 1 - odd_k + team_2

                            odd_k += 1
                            is_even = True

                        if team_1_neighbour < team_2_neighbour:
                            weeks_matches_init[Match(team_1, team_2)] = weeks_matches_init[
                                Match(team_1_neighbour, team_2_neighbour)]
                        else:
                            weeks_matches_init[Match(team_1, team_2)] = weeks_matches_init[
                                Match(team_2_neighbour, team_1_neighbour)]

            # Assign weeks for the last team matches
            for team in range(self.n_teams - 1):
                team_weeks = \
                    [weeks_matches_init[match] for match in weeks_matches_init.keys() if team in match]
                weeks_matches_init[Match(team, self.n_teams - 1)] = \
                    [week for week in range(1, self.n_teams) if week not in team_weeks][0]

            weeks_matches_init = dict(sorted(weeks_matches_init.items(), key=lambda item: item[1]))
            self.matches = np.array(list(weeks_matches_init.keys()))
            self.matches = self.matches.reshape(self.n_teams - 1, (int(self.n_teams / 2)))

    def deepcopy(self):
        # Return a deep copy of the schedule

        schedule_copy: Schedule = Schedule()
        schedule_copy.n_teams = self.n_teams
        schedule_copy.matches = self.matches.copy()

        return schedule_copy

    def matches_to_tuples(self):
        # Return the matches as a list of tuples

        return [match.to_tuple() for match in self.matches.flatten()]

    def show_as_graph(self):
        # Show the schedule as a graph

        weeks_matches_init_graph: nx.Graph = nx.Graph()
        weeks_matches_init_graph.add_edges_from(self.matches_to_tuples())

        pos = nx.spring_layout(weeks_matches_init_graph)

        nx.draw_networkx(weeks_matches_init_graph, pos, node_shape="s", node_color="none",
                         bbox=dict(facecolor="skyblue", edgecolor='none', boxstyle='round,pad=0.2'))

        nx.draw_networkx_edge_labels(
            weeks_matches_init_graph, pos,
            edge_labels={match.to_tuple(): week + 1
                         for week in range(0, self.n_teams - 1) for match in self.matches[week]},
            font_color='red'
        )

        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    schedule = Schedule(8)
    schedule.show_as_graph()
