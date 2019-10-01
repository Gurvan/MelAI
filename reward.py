
def computeRewards(observations, enemies=[1], allies=[0], damage_ratio=0.01):
    pids = enemies + allies

    deaths = processDeaths([s.players[allies[0]] for s in observations])
    height = [s.players[allies[0]].y/600 for s in observations]
 
    death_losses = [-d for d in deaths]

    return [sum(x) for x in zip(death_losses, height)]


def isDying(player):
  # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
  return player.action_state <= 0xA


def processDeaths(players):
    isded = [isDying(player) for player in players]
    loststocks = [1.0 if (next and not prev) else 0.0 for prev,next in zip(isded[:-1], isded[1:])]  
    loststocks = [0.0] + loststocks
    return loststocks
