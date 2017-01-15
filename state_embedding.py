import tensorflow as tf
import MelAPI.ssbm as ssbm

maxAction = 0x017E
numActions = 1 + maxAction
maxCharacter = 32
maxStage = 64
maxJumps = 8
action_size = len(ssbm.simpleControllerStates)


def embedEnum(enum):
    return OneHotEmbedding(len(enum))


class FloatEmbedding():
    def __init__(self, scale=None, bias=None, lower=-10.0, upper=10.0):
        self.scale = scale
        self.bias = bias
        self.lower = lower
        self.upper = upper
        self.size = 1

    def __call__(self, t):
        if t.dtype is not tf.float32:
            t = tf.cast(t, tf.float32)
        if self.bias:
            t += self.bias
        if self.lower:
            t = tf.maximum(t, self.lower)
        if self.upper:
            t = tf.minimum(t, self.upper)
        return tf.expand_dims(t, -1)


class OneHotEmbedding():
    def __init__(self, size):
        self.size = size

    def __call__(self, t):
        t = tf.cast(t, tf.int64)
        return tf.one_hot(t, self.size, 1.0, 0.0)


class ArrayEmbedding():
    def __init__(self, op, permutation):
        self.op = op
        self.permutation = permutation
        self.size = len(permutation) * op.size

    def __call__(self, array):
        embed = []
        rank = None
        for i in self.permutation:
            with tf.name_scope(str(i)):
                t = self.op(array[i])
                if rank is None:
                    rank = len(t.get_shape())
                else:
                    assert(rank == len(t.get_shape()))
                embed.append(t)
        return tf.concat_v2(embed, rank-1)


class StructEmbedding():
    def __init__(self, embedding):
        self.embedding = embedding
        self.size = 0
        for _, op in embedding:
            self.size += op.size

    def __call__(self, struct):
        embed = []
        rank = None
        for field, op in self.embedding:
            with tf.name_scope(field):
                t = op(struct[field])
                if rank is None:
                    rank = len(t.get_shape())
                else:
                    assert(rank == len(t.get_shape()))
                embed.append(t)
        return tf.concat_v2(embed, rank-1)


class DenseEmbedding():
    def __init__(self, wrapper, size, scope):
        self.wrapper = wrapper
        self.size = size
        self.scope = scope
        self.dense = lambda w: tf.contrib.layers.fully_connected(w, self.size,
                                                                 scope=self.scope,
                                                                 activation_fn=tf.nn.relu,
                                                                 reuse=None)

    def __call__(self, x):
        wrapped = self.wrapper(x)
        return self.dense(wrapped)


class PlayerEmbedding(StructEmbedding):
    def __init__(self, action_space=64):
        self.action_space = action_space
        self.xy_scale = 0.1
        self.shield_scale = 0.01
        self.speed_scale = 0.5

        embedXY = FloatEmbedding(scale=self.xy_scale)
        embedSpeed = FloatEmbedding(scale=self.speed_scale)
        embedAction = OneHotEmbedding(numActions)
        if self.action_space:
            embedAction = DenseEmbedding(embedAction, self.action_space, scope="action_space")

        playerEmbedding = [
            ('percent', FloatEmbedding(scale=0.01)),
            ('facing', embedFloat),
            ('x', embedXY),
            ('y', embedXY),
            ('action_state', embedAction),
            ('action_frame', FloatEmbedding(scale=0.02)),
            ('character', OneHotEmbedding(maxCharacter)),
            ('invulnerable', embedFloat),
            ('hitlag_frames_left', embedFloat),
            ('hitstun_frames_left', embedFloat),
            ('jumps_used', embedFloat),
            ('charging_smash', embedFloat),
            ('shield_size', FloatEmbedding(scale=self.shield_scale)),
            ('in_air', embedFloat),
            ('speed_air_x_self', embedSpeed),
            ('speed_ground_x_self', embedSpeed),
            ('speed_y_self', embedSpeed),
            ('speed_x_attack', embedSpeed),
            ('speed_y_attack', embedSpeed),
        ]

        StructEmbedding.__init__(self, playerEmbedding)


class GameEmbedding(StructEmbedding):
    def __init__(self, swap=False, player_space=64, stage_space=16, action_space=64):
        self.player_space = player_space
        embedPlayer = PlayerEmbedding(action_space)
        if self.player_space:
            embedPlayer = DenseEmbedding(embedPlayer, self.player_space, scope="player_space")

        embedStage = OneHotEmbedding(maxStage)
        if stage_space:
            embedStage = DenseEmbedding(embedStage, stage_space, scope="stage_space")

        players = [0, 1]
        if swap:
            players.reverse()

        gameEmbedding = [
            ('players', ArrayEmbedding(self.embedPlayer, players)),
            ('stage', embedStage)
        ]

        StructEmbedding.__init__(self, gameEmbedding)


embedFloat = FloatEmbedding()

stickEmbedding = [
    ('x', embedFloat),
    ('y', embedFloat)
]
embedStick = StructEmbedding(stickEmbedding)

controllerEmbedding = [
    ('button_A', embedFloat),
    ('button_B', embedFloat),
    ('button_X', embedFloat),
    ('button_Y', embedFloat),
    ('button_L', embedFloat),
    ('button_R', embedFloat),
    ('stick_MAIN', embedStick),
    ('stick_C', embedStick),
]
embedController = StructEmbedding(controllerEmbedding)

simpleAxisEmbedding = OneHotEmbedding(ssbm.axis_granularity)

simpleStickEmbedding = [
  ('x', simpleAxisEmbedding),
  ('y', simpleAxisEmbedding)
]

simpleControllerEmbedding = [
  ('button', embedEnum(ssbm.SimpleButton)),
  ('stick_MAIN', StructEmbedding(simpleStickEmbedding)),
]
