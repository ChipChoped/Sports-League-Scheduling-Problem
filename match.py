class Match(object):
    def __init__(self, team_1: int, team_2: int):
        self.team_1 = team_1
        self.team_2 = team_2

    def __repr__(self) -> str:
        return f'Match({self.team_1}, {self.team_2})'

    def __str__(self) -> str:
        return f'{self.team_1} vs {self.team_2}'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, match: object):
        return self.team_1 == match.team_1 and self.team_2 == match.team_2

    def __contains__(self, team):
        return self.team_1 == team or self.team_2 == team

    def to_tuple(self):
        return self.team_1, self.team_2

    # Check for each team if it is in conflict with the other match
    def teams_conflict(self, match: object) -> tuple[bool, bool]:
        return self.team_1 == match.team_1 or self.team_1 == match.team_2, \
               self.team_2 == match.team_1 or self.team_2 == match.team_2

    # Check if two matches have at least a team in common and so are in conflict
    def conflict(self, match: object) -> bool:
        return self.team_1 == match.team_1 or self.team_1 == match.team_2 \
                or self.team_2 == match.team_1 or self.team_2 == match.team_2
