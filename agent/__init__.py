from .dqn import AgentSolver, Agent, AgentVec, MultiStepReplayBuffer, ReplayBuffer


def make_agent(solver: AgentSolver):
    """Factory: dispatch on solver.qnet_name.

    'multistep_vec' -> per-class vectorized agent (WT5 Path B); otherwise the scalar
    Agent (master behavior, covers A2c/A3c/A5c/A6c ablations).
    """
    if getattr(solver, 'qnet_name', 'multistep').lower() == 'multistep_vec':
        return AgentVec(solver)
    return Agent(solver)
