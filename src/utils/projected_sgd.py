import torch
from torch.func import jacrev
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def projected_sgd_step(
    model,
    loss_fn,  # callable: loss_fn(model, batch) -> scalar loss
    batch,
    constraint_fn,  # callable: constraint_fn(theta_flat) -> (m,) tensor
    lr=1e-3,
    n_project_iters=10,
    damping=1e-6,
):
    """
    One projected-SGD step for:
        minimize L(theta) subject to constraint_fn(theta) = 0

    Parameters
    ----------
    model : nn.Module
    loss_fn : function
        Returns a scalar loss.
    batch : any
        Passed into loss_fn.
    constraint_fn : function
        Input: flat parameter vector theta of shape (p,)
        Output: constraint vector f(theta) of shape (m,)
    lr : float
    n_project_iters : int
        Number of Newton-style projection iterations after tangent step.
    damping : float
        Tikhonov damping for solving systems involving J J^T.
    """

    # ---- 1) ordinary gradient of the training loss ----
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    loss.backward()

    with torch.no_grad():
        theta = parameters_to_vector(model.parameters()).detach()

        grads = []
        for p in model.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p).reshape(-1))
            else:
                grads.append(p.grad.reshape(-1))
        g = torch.cat(grads).detach()

    # ---- 2) Jacobian J_f(theta) ----
    # J has shape (m, p) if constraint_fn(theta) has shape (m,)
    J = jacrev(constraint_fn)(theta)  # exact Jacobian wrt flat parameter vector

    # Make sure f(theta) is a 1D vector
    f_theta = constraint_fn(theta)
    if f_theta.ndim == 0:
        f_theta = f_theta.unsqueeze(0)
        J = J.unsqueeze(0)

    # ---- 3) tangent projection of gradient ----
    # g_tan = g - J^T (J J^T)^(-1) J g
    JJt = J @ J.T
    m = JJt.shape[0]
    JJt_reg = JJt + damping * torch.eye(m, device=JJt.device, dtype=JJt.dtype)

    rhs = J @ g
    alpha = torch.linalg.solve(JJt_reg, rhs)
    g_tan = g - J.T @ alpha

    # ---- 4) tangent step ----
    theta_new = theta - lr * g_tan

    # ---- 5) Newton-style projection back to manifold ----
    # theta <- theta - J(theta)^T (J J^T)^(-1) f(theta)
    for _ in range(n_project_iters):
        f_val = constraint_fn(theta_new)
        if f_val.ndim == 0:
            f_val = f_val.unsqueeze(0)

        J_new = jacrev(constraint_fn)(theta_new)
        if f_val.numel() == 1 and J_new.ndim == 1:
            J_new = J_new.unsqueeze(0)

        JJt_new = J_new @ J_new.T
        m = JJt_new.shape[0]
        JJt_new_reg = JJt_new + damping * torch.eye(
            m, device=JJt_new.device, dtype=JJt_new.dtype
        )

        delta = J_new.T @ torch.linalg.solve(JJt_new_reg, f_val)
        theta_new = theta_new - delta

    # ---- 6) write parameters back ----
    with torch.no_grad():
        vector_to_parameters(theta_new, model.parameters())

    return {
        "loss": loss.item(),
        "constraint_norm_before": constraint_fn(theta).norm().item(),
        "constraint_norm_after": constraint_fn(theta_new).norm().item(),
        "tangent_grad_norm": g_tan.norm().item(),
    }
