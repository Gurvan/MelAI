import numpy as np

dyingActions = set(range(0xA))
lyingActions = set([0xB7, 0xB8, 0xB9, 0xBF, 0xC0, 0xC1])


def isDyingAction(action_state):
    return action_state in dyingActions


def isDying(player):
    return isDyingAction(player.action_state)


def isLyingAction(action_state):
    return action_state in lyingActions


def isLying(player):
    return isLyingActions(player.action_state)


def processDeath(deaths):
    return np.array(util.zipWith(lambda prev, next: float((not prev) and next), deaths, deaths[1:]))


def processDamages(percents):
    return np.array(util.zipWith(lambda prev, next: max(next-prev, 0), percents, percents[1:]))


def computeRewards(state_actions, opponents=[0], allies=[1], damage_ratio=0.01, lying_ratio=0.002):
    players = opponents + allies
    deaths = {p: processDeaths([isDying(sa.state.players[p]) for sa in state_actions]) for p in players}
    damages = {p: processDamages([sa.state.players[p].percent for sa in state_actions]) for p in players}

    lying = {p: [isLying(sa.state.players[p]) for sa in state_actions] for p in players}

    hitstuns = {p: [sa.state.players[p].hitstun_frames_left for sa in state_actions] for p in players}

    ajustedDamages = {p: [] for p in players}
    for p in players:
        ajustedDamages[p] = [damage + 10.0/hitstun * (damage > 0) for damage, hitstun in zip(damages[p], hitstuns[p])]

    losses = {p: deaths[p] + damage_ratio * ajustedDamages[p] + lying_ratio * lying[p] for p in players}

    return sum(losses[p] for p in enemies) - sum(losses[p] for p in allies)
