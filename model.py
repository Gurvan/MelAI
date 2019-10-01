import torch
import torch.nn as nn
import torch.nn.functional as F

class DuellingDQN(torch.nn.Module):
    def __init__(self, state_shape, action_dim):
        super(DuellingDQN, self).__init__()
        self.input_shape = state_shape
        self.action_dim = action_dim
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(state_shape[0], 64, 8, stride=4),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 4, stride=2),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1),
                                          torch.nn.ReLU())
        self.value_stream_layer = torch.nn.Sequential(torch.nn.Linear( 64 * 7 * 7, 512),
                                                      torch.nn.ReLU())
        self.advantage_stream_layer = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                                          torch.nn.ReLU())
        self.value = torch.nn.Linear(512, 1)
        self.advantage = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        #assert x.shape == self.input_shape, "Input shape should be:" + str(self.input_shape) + "Got:" + str(x.shape)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = x.view(-1, 64 * 7 * 7)
        value = self.value(self.value_stream_layer(x))
        advantage = self.advantage(self.advantage_stream_layer(x))
        action_value = value + (advantage - (1/self.action_dim) * advantage.sum() )
        return value, advantage, action_value


class Actor(torch.nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.game_layer = GameLayer(64)
        self.policy = nn.Linear(64, action_dim)
        

    def forward(self, x):
        x = gamesTensors([x])
        x = self.game_layer(*x)
        p = self.policy(x)
        logit = F.log_softmax(p, 1)
        return torch.distributions.Categorical(logits=logit).sample()[0]





class PlayerLayer(torch.nn.Module):
    def __init__(self, output_dim, num_characters = 32, num_action_states = 0x017F, state_size = 10):
        super(PlayerLayer, self).__init__()
        self.character_dim = 10
        self.action_state_dim = 32
        self.hidden_dim = 64
        self.character_layer = nn.Sequential(
            nn.Linear(num_characters, 32),
            nn.Linear(32, self.character_dim)
        )
        self.action_state_layer = nn.Sequential(
            nn.Linear(num_action_states, 64),
            nn.Linear(64, self.action_state_dim)
        )
        self.player_layer = nn.Sequential(
            nn.Linear(self.character_dim + self.action_state_dim + state_size, self.hidden_dim),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, x_character, x_action_state, x_state):
        x_character = self.character_layer(x_character)
        x_action_state = self.action_state_layer(x_action_state)
        x = self.player_layer(torch.cat([x_character, x_action_state, x_state], 1))
        return x


class GameLayer(torch.nn.Module):
    def __init__(self, output_dim, player_output_dim = 29, num_stages = 32):
        super(GameLayer, self).__init__()
        self.stage_dim = 6
        self.action_state_dim = 32
        self.hidden_dim = 128
        self.stage_layer = nn.Sequential(
            nn.Linear(num_stages, 32),
            nn.Linear(32, self.stage_dim)
        )
        self.player0_layer = PlayerLayer(player_output_dim)
        self.player1_layer = PlayerLayer(player_output_dim)

        self.game_layer = nn.Sequential(
            nn.Linear(2 * player_output_dim + self.stage_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, x_player0, x_player1, x_stage):
        x_player0 = self.player0_layer(*x_player0)
        x_player1 = self.player1_layer(*x_player1)
        x_stage = self.stage_layer(x_stage)
        x = self.game_layer(torch.cat([x_player0, x_player1, x_stage], 1))
        return x



def unpack(a):
    return list(zip(*a))


def gamesTensors(x, device='cpu'):
    p0,p1,stages = unpack(x)
    p0 = tuple((torch.tensor(a).to(device) for a in unpack(p0)))
    p1 = tuple((torch.tensor(a).to(device) for a in unpack(p1)))
    stages = torch.tensor(stages).to(device)

    return p0, p1, stages

