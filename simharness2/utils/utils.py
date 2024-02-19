from typing import TYPE_CHECKING
import logging
from itertools import chain

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.algorithm import Algorithm


logger = logging.getLogger(__name__)


def validate_evaluation_config(algo_cfg: "AlgorithmConfig"):
    """Ensure the configuration of evaluation workers is valid.

    Arguments:
        algo_cfg: The rllib `AlgorithmConfig` instance.

    Raises:
        ValueError: If the evaluation duration unit is not `episodes`.
        ValueError: If the number of evaluation workers is 0.
        ValueError: If the evaluation duration is greater than the number of
            evaluation workers.
        ValueError: If the evaluation `WorkerSet` uses a local worker.
        ValueError: If the evaluation `WorkerSet` uses sub-environments (is vectorized).

    """
    num_eval_workers = algo_cfg.evaluation_num_workers
    eval_duration = algo_cfg.evaluation_duration
    eval_duration_unit = algo_cfg.evaluation_duration_unit

    use_train_local_worker = algo_cfg.create_env_on_local_worker
    use_eval_local_worker = algo_cfg.evaluation_config.get("create_env_on_local_worker")
    envs_per_train_worker = algo_cfg.num_envs_per_worker
    envs_per_eval_worker = algo_cfg.evaluation_config.get("num_envs_per_worker")

    if eval_duration_unit != "episodes":
        msg = "The `evaluation_duration_unit` must be set to `episodes`."
        raise ValueError(msg)
    # TODO: Handle `num_eval_workers == 0` edge case.
    elif num_eval_workers == 0:
        msg = "The `evaluation_num_workers` must be greater than 0."
        raise ValueError(msg)
    # TODO: Handle `eval_duration` greater than `evaluation_num_workers` edge case.
    elif eval_duration / num_eval_workers > 1:
        msg = "The `evaluation_duration` cannot be greater than `evaluation_num_workers`."
    # The logic gets weird if eval envs use a local worker; don't allow this for now.
    # TODO: Handle `use_eval_local_worker` edge case.
    elif use_train_local_worker and (
        use_eval_local_worker is None or use_eval_local_worker
    ):
        # Error if value is unspecified in evaluation config or set to True.
        if use_eval_local_worker is None:
            msg = (
                "The `create_env_on_local_worker` must be set to `False` for evaluation "
                "environments. This is unspecified, so we inherit the training workers "
                f"value for `create_env_on_local_worker`: {use_train_local_worker}. To "
                "override this behavior, set "
                "`evaluation.evaluation_config.create_env_on_local_worker` to `False`."
            )
        else:
            msg = (
                "The `create_env_on_local_worker` must be set to `False` for evaluation "
                f"environments, got: {use_eval_local_worker}. Update the value of "
                "`evaluation.evaluation_config.create_env_on_local_worker` to `False`."
            )
        raise ValueError(msg)
    # The logic gets weird if eval envs use sub-envs; don't allow this for now.
    # TODO: Handle `envs_per_eval_worker > 1` edge case.
    elif envs_per_train_worker > 1 and (
        envs_per_eval_worker is None or envs_per_eval_worker > 1
    ):
        # Error if unspecified in evaluation config or set to > 1.
        if envs_per_eval_worker is None:
            msg = (
                "The `num_envs_per_worker` must be set to 1 for evaluation "
                "workers. This is unspecified, so we inherit the training workers value "
                f"for `num_envs_per_worker`: {envs_per_train_worker}. To override this "
                "behavior, set `evaluation.evaluation_config.num_envs_per_worker` to 1."
            )
        else:
            msg = (
                "The `num_envs_per_worker` must be set to 1 for evaluation "
                f"workers, got: {envs_per_eval_worker}. Update the value of "
                "`evaluation.evaluation_config.num_envs_per_worker` to 1."
            )
        raise ValueError(msg)


def validate_rollouts_config(algo_cfg: "AlgorithmConfig"):
    """Ensure the configuration of rollout workers is valid."""
    num_train_workers = algo_cfg.num_rollout_workers
    use_train_local_worker = algo_cfg.create_env_on_local_worker

    if num_train_workers > 0 and use_train_local_worker:
        msg = (
            "When num_rollout_workers > 0, the driver (local_worker; worker-idx=0) does "
            "not need an environment. This is because it doesn't have to sample (done "
            "by remote_workers; worker_indices > 0) nor evaluate (done by evaluation "
            "workers)."
        )
        raise ValueError(msg)


def get_total_training_envs(algorithm: "Algorithm") -> int:
    """Return the total number of training envs.

    The `foreach_env` call will return a nested list. Each index in `train_envs`
    corresponds to each `worker_index` in `algorithm.workers`. Each index contains a list
    of sub-environments contained within the respective worker. The `*train_envs` syntax
    expands the nested list into individual args to `chain()`, which concatenates them
    together into one long list, and `len()` returns its length.

    Arguments:
        algorithm: The rllib `Algorithm` instance.
    """
    train_envs = algorithm.workers.foreach_env(lambda env: env)
    return len(list(chain(*train_envs)))


def get_total_evaluation_envs(algorithm: "Algorithm") -> int:
    """Return the total number of evaluation envs.

    The `foreach_env` call will return a nested list. Each index in `eval_envs`
    corresponds to each `worker_index` in `algorithm.evaluation_workers`. Each index
    contains a list of sub-environments contained within the respective worker. The
    `*eval_envs` syntax expands the nested list into individual args to `chain()`, which
    concatenates them together into one long list, and `len()` returns its length.

    NOTE: The number of evaluation envs is equal to the number of evaluation
    workers, enforced by the current implementation of `validate_evaluation_config`.
    Once callbacks, such as `InitializeSimfire`, are updated to handle
    `eval_duration / num_eval_workers > 1`, this method will need to be updated.
    """
    eval_envs = algorithm.evaluation_workers.foreach_env(lambda env: env)
    return len(list(chain(*eval_envs)))
