from gym.envs.registration import register

register(
    id='Slime-v0',
    entry_point='slime_environments.environments:Slime',
    max_episode_steps=10000,  # DOC The keyword argument max_episode_steps=300 will ensure that GridWorld environments that are instantiated via gym.make will be wrapped in a TimeLimit wrapper (https://www.gymlibrary.dev/content/environment_creation/#registering-envs)
    nondeterministic=True  # DOC seeding not supported atm
)
