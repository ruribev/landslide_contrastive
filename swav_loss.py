import torch
import torch.nn.functional as F

def distributed_sinkhorn(Q, n_iters=3, epsilon=1e-6):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        K, B = Q.shape
        u = torch.zeros(K, device=Q.device)
        r = torch.ones(K, device=Q.device) / K
        c = torch.ones(B, device=Q.device) / B
        for _ in range(n_iters):
            u = torch.sum(Q, dim=1)
            Q *= (r / (u + epsilon)).unsqueeze(1)
            Q *= (c / (torch.sum(Q, dim=0) + epsilon)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def swav_loss_func(logits, temperature=0.1, sinkhorn_iters=3):
    # logits: [N_views, K]
    # 1. Softmax para similitud
    scores = logits / temperature
    probs = F.softmax(scores, dim=1)
    # 2. Sinkhorn para obtener asignaciones balanceadas
    Q = distributed_sinkhorn(probs.t(), n_iters=sinkhorn_iters)  # [N_views, K]
    # 3. Swapped prediction loss (cross-entropy entre vistas)
    loss = 0
    n_views = logits.shape[0]
    for i in range(0, n_views, 2):
        p1, p2 = probs[i], probs[i+1]
        q1, q2 = Q[i], Q[i+1]
        loss += - (q2 * torch.log(p1 + 1e-6)).sum() / K
        loss += - (q1 * torch.log(p2 + 1e-6)).sum() / K
    loss /= (n_views // 2)
    return loss
