import numpy as np
import MelAPI.util as util

dyingActions = set(range(0xA))
lyingActions = set([0xB7, 0xB8, 0xB9, 0xBF, 0xC0, 0xC1])


def isDyingAction(action_state):
    return action_state in dyingActions


def isDying(player):
    return isDyingAction(player.action_state)


def isLyingAction(action_state):
    return action_state in lyingActions


def isLying(player):
    return isLyingAction(player.action_state)


def processDeaths(deaths):
    return float(deaths[1] and not deaths[0])
    # return np.array(util.zipWith(lambda prev, next: float((not prev) and next), [deaths[0]], deaths[1:]))


def processDamages(percents):
    return max(percents[1] - percents[0], 0)
    # return np.array(util.zipWith(lambda prev, next: max(next-prev, 0), [percents[0]], percents[1:]))


def computeRewards(state_actions, opponents=[0], allies=[1], damage_ratio=0.01, lying_ratio=0.002):
    players = opponents + allies
    deaths = {p: processDeaths([isDying(sa.players[p]) for sa in state_actions]) for p in players}
    damages = {p: processDamages([sa.players[p].percent for sa in state_actions]) for p in players}

    lying = {p: isLying(state_actions[0].players[p]) for p in players}

    hitstuns = {p:  state_actions[0].players[p].hitstun_frames_left for p in players}

    ajustedDamages = {p: 0 for p in players}
    for p in players:
        if (damages[p] > 0) and (hitstuns[p] >= 1):
            ajustedDamages[p] = damages[p] + 10.0/(1.0+hitstuns[p])
        else:
            ajustedDamages[p] = damages[p]

    losses = {p: deaths[p] + damage_ratio * ajustedDamages[p] + lying_ratio * lying[p] for p in players}

    return np.sum(losses[p] for p in opponents) - np.sum(losses[p] for p in allies)
