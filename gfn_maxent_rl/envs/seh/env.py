import itertools
import numpy as np
import gym
import math
import operator as op

from gym.spaces import Dict, Box, Discrete, MultiDiscrete, Tuple
from functools import reduce
from numpy.random import default_rng
import torch
import torch_geometric as tg
import networkx as nx

import sys
import os
sys.path.append(os.getcwd())

from gfn_maxent_rl.envs.errors import PermutationEnvironmentError, StatesEnumerationError
from gfn_maxent_rl.envs.seh.policy import uniform_log_policy
from gfn_maxent_rl.envs.seh.rewards import RewardProxy

# from gflownet.algo.graph_sampling import GraphSampler
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphActionType, GraphBuildingEnv, Graph
from gflownet.models.bengio2021flow import FRAGMENTS


def list_of_dict_to_dict_of_list(lst):
    if not lst:
        return {}
    return {k: [l[k] for l in lst] for k in lst[0].keys()}


def observations_to_states(observations):
    # each g contains a tuple (adjacency, node_attr)
    graphs = [nx.from_numpy_array(g[0], edge_attr=None, create_using=Graph) for g in observations['graph']]
    # set node-attributes
    for i in range(len(graphs)):
        attr = observations['graph'][i][1]
        for j in range(len(attr)):
            graphs[i].nodes[j]['v'] = attr[j]
    return graphs


class SEHEnvironment(gym.vector.VectorEnv):

    def __init__(self, num_envs, seed=0):
        self.num_envs = num_envs
        fragments = FRAGMENTS

        ctx = FragMolBuildingEnvContext(max_frags=9, num_cond_dim=32, fragments=fragments)
        env = GraphBuildingEnv(allow_add_edge=True, allow_node_attr=True, allow_edge_attr=True)
        rng = np.random.RandomState(seed=seed)
        self.reward_model = RewardProxy() # computes the reward

        self.ctx = ctx
        self.env = env
        self.time_limit = 128
        self.max_nodes = 9
        self.n_fragments = len(fragments)
        self.sanitize_samples = True
        self.correct_idempotent = True

        self.node_shape = (ctx.num_new_node_values, self.n_fragments)
        self.set_edge_shape = (ctx.num_new_node_values, ctx.num_edge_attr_logits)

        # initialize env vars
        self.states = [None for i in range(self.num_envs)]
        self.torch_graphs = [None for i in range(self.num_envs)]
        self.timestep = np.zeros(self.num_envs, dtype=np.int32)

        # [GraphActionType.Stop, GraphActionType.AddNode, GraphActionType.SetEdgeAttr], , num_edge_attr_logits
        # action_space = Tuple(
        #     MultiDiscrete([2]),
        #     MultiDiscrete([ctx.num_new_node_values, len(fragments)]),
        #     MultiDiscrete([ctx.num_new_node_values, ctx.num_edge_attr_logits])
        #     )
        action_space = Discrete(1 + ctx.num_new_node_values*self.n_fragments + ctx.num_new_node_values*ctx.num_edge_attr_logits)
        observation_space = Dict({
            'nodes': Box(low=-np.inf, high=np.inf, shape=self.node_shape),
            'mask': Box(low=0, high=1, shape=(action_space.n,))})

        super().__init__(num_envs, observation_space, action_space)

    def _action_mask(self, torch_graph):
        """
        There are 3 action-types:
        0. stop
        1. add-node
        2. set-edge

        - The stop-action is only legal when current solution is valid
        - For the add-node action, you have to set 2 things:
            a) the position where you want to add the node.
               This depends on the current number of nodes of the graph.
               So for a graph with 3 nodes, you can only add a node to any of the 3 existing nodes.
               ?? So mask node-indexes where num-nodes < node-indexes < max-nodes.
            b) the type of node you want to add. This is equal to the number of provided fragments.
               ?? all types are always legal
        - For the set-edge action, you have:
            a) the edge-index where you set the attribute.
               There are #nodes-1 edges
            b) ?? the stem-index related with the node-type that is connected by this edge
               use stem-index to connect to src, and stem-index + num_stem_acts to connect to dest
        """
        stop_mask = np.ones(1, dtype=np.uint8)*torch_graph.stop_mask.item()
        add_mask = np.zeros((self.ctx.num_new_node_values, self.n_fragments), dtype=np.uint8)
        add_mask[:len(torch_graph.add_node_mask)] = torch_graph.add_node_mask.numpy()
        set_mask = np.zeros((self.ctx.num_new_node_values, self.ctx.num_edge_attr_logits), dtype=np.uint8)
        set_mask[:len(torch_graph.set_edge_attr_mask)] = torch_graph.set_edge_attr_mask.numpy()
        return np.concatenate((stop_mask, add_mask.flatten(), set_mask.flatten())).astype(np.int8)
    
    def index_to_action(self, a_i):
        # stop-action
        if a_i == 0:
            return (0, 0, 0)
        # add-node action
        elif a_i < 1+self.ctx.num_new_node_values*self.n_fragments:
            a_i = a_i - 1
            return (1,) + np.unravel_index(a_i, (self.ctx.num_new_node_values, self.n_fragments))
        # set edge action
        else:
            a_i = a_i - (1+self.ctx.num_new_node_values*self.n_fragments)
            return (2,) + np.unravel_index(a_i, (self.ctx.num_new_node_values, self.ctx.num_edge_attr_logits))
        
    def observations(self, list_of_obs):
        obs = {}
        # graph is a list of np matrices of unequal shapes, keep it like that
        obs['graph'] = [o['graph'] for o in list_of_obs]
        # nodes is a list of np matrices of equal shape, make it a np array
        obs['nodes'] = np.stack([o['nodes'] for o in list_of_obs], axis=0)
        # same for mask
        obs['mask'] = np.stack([o['mask'] for o in list_of_obs], axis=0)
        return obs
    
    def to_observation(self, env_i):
        torch_graph = self.torch_graphs[env_i]
        # nodes
        x = torch_graph.x.numpy()
        nodes = np.empty(self.node_shape)
        nodes[:len(x)] = x[:, :nodes.shape[1]]
        # graph
        g = self.states[env_i]
        graph = nx.to_numpy_array(g)
        # `v` attribute holds fragment index
        attr = [g.nodes[n]['v'] for n in g.nodes]
        return {
            'graph': (graph, attr),
            'nodes': nodes,
            'mask': self._action_mask(torch_graph)
        }

    def single_reset(self, env_i, *, seed=None, options=None):
        state = self.env.new()

        self.states[env_i] = state
        self.torch_graphs[env_i] = self.ctx.graph_to_Data(self.states[env_i])
        self.timestep[env_i] = 0

        return self.to_observation(env_i), {'is_valid': True}

    def reset(self, *, seed=None, options=None):
        resets = [self.single_reset(i) for i in range(self.num_envs)]
        observations, info = list(zip(*resets))
        return self.observations(observations), list_of_dict_to_dict_of_list(info)
    
    def single_step(self, env_i, action):
        """
        For now, looks like action is a multi-discrete, of <int, int, int>
        computes binding prob using proxy model.
        Provide log-prob of valid solutions as reward (so it is already log-reward for gfn)
        """
        action = self.index_to_action(action)
        state, torch_graph = self.states[env_i], self.torch_graphs[env_i]
        reward = 0.
        graph_action = self.ctx.aidx_to_GraphAction(torch_graph, action)
        
        truncated = self.timestep[env_i] == self.time_limit
        # TODO technically done and truncated should be handled separately
        done = (graph_action.action is GraphActionType.Stop) or truncated
        is_valid = True
        # only add fragments if the action is not a stop-action
        if not done:
            # this raises an assertion-error if the action is illegal
            try:
                state = self.env.step(state, graph_action)
                torch_graph = self.ctx.graph_to_Data(state)
            except AssertionError as e:
                is_valid = False
                done = True
            # update internal state
            self.timestep[env_i] += 1
            self.states[env_i] = state
            self.torch_graphs[env_i] = torch_graph
            observation = self.to_observation(env_i)
        # don't use else here, as done can be modified in the previous if-test
        if done:
            # check if the graph is sane (e.g. RDKit can
            # construct a molecule from it) otherwise
            # treat the done action as illegal
            if self.sanitize_samples and not self.ctx.is_sane(state):
                is_valid = False
            else:
                # if it is a valid graph, compute the reward
                # expects a list of inputs, make a list of 1 element and return the first result only
                reward = self.reward_model.compute_flat_reward([self.ctx.graph_to_mol(state)]).item()
            # done, so reset state
            observation, _ = self.single_reset(env_i)

        return observation, reward, done, truncated, {'is_valid': is_valid}
        
    def step(self, actions):
        steps = [self.single_step(i, a) for i, a in enumerate(actions)]
        observation, reward, done, truncated, info = list(zip(*steps))
        
        return (
            self.observations(observation),
            np.array(reward),
            np.array(done),
            np.array(truncated),
            list_of_dict_to_dict_of_list(info)
        )
    
    # Properties & methods to interact with the replay buffer

    @property
    def observation_dtype(self):
        # max-sized graph
        return np.dtype([
            ('nodes', np.float32, self.node_shape),
            ('mask', np.uint8, (self.single_action_space.n,)),
        ])

    @property
    def max_length(self):
        return self.time_limit

    def encode(self, observations):
        batch_size = len(observations['mask'])
        encoded = np.empty((batch_size,), dtype=self.observation_dtype)
        for k in self.observation_dtype.names:
            encoded[k] = observations[k]
        return encoded

    def decode(self, observations):
        decoded = {k: observations[k] for k in self.observation_dtype.names}
        return decoded
    
    @property
    def observation_sequence_dtype(self):
        return self.observation_dtype

    def encode_sequence(self, observations):
        return self.encode(observations)

    def decode_sequence(self, samples):
        return self.decode(samples)
    
    # Method to interact with the algorithm (uniform sampling of action)

    def uniform_log_policy(self, observations):
        return uniform_log_policy(observations['mask'])

    def num_parents(self, observations):
        graphs = observations_to_states(observations)
        n_back = [self.env.count_backward_transitions(g, check_idempotent=self.correct_idempotent) for g in graphs]
        return np.array(n_back)

    def action_mask(self, observations):
        return observations['mask']
    
    # Method for evaluation

    def all_states_batch_iterator(self, batch_size, terminating=False):
        raise StatesEnumerationError('Impossible to enumerate all the '
            'states of `PhyloTreeEnvironment`.')
    
    def log_reward(self, observations):
        states = observations_to_states(observations)
        reward = self.reward_model.compute_flat_reward([self.ctx.graph_to_mol(s) for s in states])
        reward = reward.numpy().flatten()
        return np.log(reward)
    
    @property
    def mdp_state_graph(self):
        raise StatesEnumerationError('Impossible to enumerate all the '
            'states of `PhyloTreeEnvironment`.')

    def observation_to_key(self, observations):
        states = observations_to_states(observations)
        return states

    def key_batch_iterator(self, keys, batch_size):
        for index in range(0, len(keys), batch_size):
            yield (keys[index:index + batch_size], self.time_limit)

    def key_to_action_mask(self, keys):
        raise PermutationEnvironmentError('The environment does not generate '
            'objects as permutations of actions.')

    def backward_sample_trajectories(
            self,
            keys,
            num_trajectories,
            max_length=None,
            blacklist=None,
            rng=default_rng(),
            max_retries=10
    ):
        if blacklist is not None:
            raise NotImplementedError('Argument `blacklist` must be `None`.')

        trajectories = np.full((len(keys), num_trajectories, self.max_length), -1, dtype=np.int_)
        log_pB = np.zeros((len(keys), num_trajectories), dtype=np.float_)

        for k, n in itertools.product(range(len(keys)), range(num_trajectories)):
            state = keys[k]
            for t in range(self.time_limit):
                # sample random backward action
                remove_node = np.zeros(self.max_nodes, dtype=np.float32)
                remove_node[:len(state.remove_node_mask)] = state.remove_node_mask.flatten()
                remove_edge_attr = np.zeros(self.ctx.num_new_node_values*2, dtype=np.float32)
                remove_edge_attr[:np.prod(state.remove_edge_attr_mask.shape)] = state.remove_edge_attr_mask.flatten()
                action_probs = np.concatenate((remove_node, remove_edge_attr))
                # sample based on index
                a_i = np.random.multinomial(1, action_probs/np.sum(action_probs))
                # convert index to backward action
                if a_i < self.max_nodes:
                    action = (0, a_i, 0)
                else:
                    action = a_i - self.max_nodes
                    action = (1,) + np.unravel_index(action, (self.ctx.num_new_node_values, 2))
                graph_action = self.ctx.aidx_to_GraphAction(state, action, fwd=False)
                state = self.env.step(state, graph_action)
                # log results
                trajectories[k, n, t] = a_i
                log_pB[k, n] += np.log(1./np.sum(action_probs))
                if not len(state):
                    break

        return (trajectories, log_pB)

    

if __name__ == '__main__':
    env = SEHEnvironment(10)

    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample(mask=tuple([i for i in obs['mask']]))
        n_obs, reward, done, truncated, info = env.step(action)

        for i in range(env.num_envs):
            if done[i]:
                print(obs['graph'][i], 'valid: ', info['is_valid'][i], f'reward {reward[i]}')
                n_back = [env.env.count_backward_transitions(g, check_idempotent=env.correct_idempotent) for g in env.states]
                n_back2 = env.num_parents(n_obs)
                breakpoint()

        obs = n_obs

