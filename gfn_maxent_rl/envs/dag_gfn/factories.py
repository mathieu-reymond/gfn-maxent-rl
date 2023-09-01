from gfn_maxent_rl.envs.dag_gfn.scores import ZeroScore
from gfn_maxent_rl.envs.dag_gfn.graph_priors import UniformPrior
from gfn_maxent_rl.envs.dag_gfn.env import DAGEnvironment
from gfn_maxent_rl.envs.dag_gfn.base import JointModel


def get_marginal_likelihood(name, data, **kwargs):
    scores = {
        'zero': ZeroScore,
        # 'lingauss': LinearGaussianScorer,
        # 'bge': BGeScorer,
    }
    if name not in scores:
        valid_scorers = ', '.join(scores.keys())
        raise ValueError(f'Unknown score (marginal likelihood): {name}. Must be one of {{{valid_scorers}}}.')
    return scores[name](data, **kwargs)


def get_graph_prior(name, num_variables, **kwargs):
    priors = {
        'uniform': UniformPrior,
    }
    if name not in priors:
        valid_priors = ', '.join(priors.keys())
        raise ValueError(f'Unknown graph prior: {name}. Must be one of {{{valid_priors}}}.')
    return priors[name](num_variables, **kwargs)


def get_dag_gfn_env(
    data,
    prior_name,
    scorer_name,
    num_envs=1,
    prior_kwargs={},
    scorer_kwargs={},
):
    # Get the graph prior & marginal likelihood for reward computation
    num_variables = data.shape[1]
    graph_prior = get_graph_prior(prior_name, num_variables, **prior_kwargs)
    marginal_likelihood = get_marginal_likelihood(scorer_name, data, **scorer_kwargs)
    joint_model = JointModel(
        graph_prior=graph_prior,
        marginal_likelihood=marginal_likelihood
    )

    # Create the environment
    env = DAGEnvironment(num_envs=num_envs, joint_model=joint_model)

    return env