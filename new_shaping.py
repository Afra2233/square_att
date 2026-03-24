attack_cfg = SquareAttackConfig(
    eps=8 / 255,
    n_iters=200,
    eot_M=16,
    min_square=1,
    max_square=64,
    seed=0,
)

aaa_base_cfg = AAAConfig(
    temperature=1.0,
    tau=6.0,
    beta=5.0,
    kappa=100,
    lr=0.1,
    alpha_linear=1.0,
    alpha_sine=0.7,
)