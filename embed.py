maxAction = 0x017E
numActions = 1 + maxAction

numCharacters = 32 # should be large enough?
numStages = 32
maxJumps = 8


def oneHot(x, n):
    y = n * [0.0]
    y[x] = 1.0
    return y

def embedPlayer(player_state, flat=True):
    percent = player_state.percent/100.0
    facing = player_state.facing
    x = player_state.x/10.0
    y = player_state.y/10.0
    action_state = oneHot(player_state.action_state, numActions)
    action_frame = player_state.action_frame/50.0
    character = oneHot(player_state.character, numCharacters)
    invulnerable = 1.0 if player_state.invulnerable else 0
    hitlag_frames_left = player_state.hitlag_frames_left/10.0
    hitstun_frames_left = player_state.hitstun_frames_left/10.0
    #jumps_used = float(player_state.jumps_used)
    #charging_smash = 1.0 if player_state.charging_smash else 0.0
    shield_size = player_state.shield_size/100.0
    in_air = 1.0 if player_state.in_air else 0.0

    if flat:
        return [
            character,
            action_state,
            [
                percent,
                facing,
                x, y,
                action_frame,
                invulnerable,
                hitlag_frames_left,
                hitstun_frames_left,
                shield_size,
                in_air
                ]
            ]

    return dict({
        'character': character,
        'action_state': action_state,
        'state': [
            percent,
            facing,
            x, y,
            action_frame,
            invulnerable,
            hitlag_frames_left,
            hitstun_frames_left,
            shield_size,
            in_air]
        })


def embedGame(game_state, flat=True):
    player0 = embedPlayer(game_state.players[0], flat)
    player1 = embedPlayer(game_state.players[1], flat)
    stage = oneHot(game_state.stage, numStages)

    if flat:
        return [
            player0,
            player1,
            stage,
        ]
    return dict({
        'player0': player0,
        'player1': player1,
        'stage': stage,
    })
