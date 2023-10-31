# Sports League Scheduling Problem

## How to run the program

A version of python 3.10 or higher is required.

```bash
python3 sports_league_scheduling.py <number of teams> <number of maximum iterations> <tabu list size> <number of tests (optional)>
```

### Example
The following command will run the program with 8 teams, 1000 maximum iterations and a tabu list of size 32 for 10 tests:

```bash
python3 sports_league_scheduling.py 8 1000 32 10
```

## See the initialization process
You can visualize the schedule initialization graph with the following command:

```bash
python3 schedule.py <number of teams>
```