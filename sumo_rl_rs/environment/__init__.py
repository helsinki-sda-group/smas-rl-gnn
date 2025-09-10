"""SUMO Environment for Ride-Sharing."""

from gymnasium.envs.registration import register


register(
    id="sumo-rl-rs-v0",
    entry_point="sumo_rl_rs.environment.env:SumoEnvironment",
    kwargs={},
)

from .ridepool_rt_env import RidepoolRTEnv

__all__ = ["RidepoolRTEnv"]