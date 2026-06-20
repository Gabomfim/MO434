import torch
import torch.nn as nn
import torch.nn.functional as F
from metric.utils import pdist

__all__ = ['L1Triplet', 'L2Triplet', 'ContrastiveLoss', 'RkdDistance', 'RKdAngle', 'HardDarkRank',
           'RkdQuadrupletSum']


class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p==2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                     margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)


class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_loss = (F.pairwise_distance(anchor_embed, positive_embed, p=2)).pow(2)
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed, negative_embed, p=2)).clamp(min=0).pow(2)

        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()


class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class FitNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, student, teacher):
        if student.dim() == 2:
            student = student.unsqueeze(2).unsqueeze(3)
            teacher = teacher.unsqueeze(2).unsqueeze(3)

        return (self.transform(student) - teacher).pow(2).mean()


class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


class RkdQuadrupletSum(nn.Module):
    """
    Relational KD loss on all 4-sample sets in a minibatch.

    For each unordered set {A, B, C, D}, compute the sum of all 6 pairwise
    Euclidean distances in embedding space. Then normalize teacher and student
    set-wise sums by each network's mean (like RKD distance normalization) and
    minimize the smooth L1 difference.
    """

    # Cache dos indices de combinacao por (n, device): so dependem do batch.
    _comb_cache = {}

    def _combinations(self, n, device):
        key = (n, device)
        comb = self._comb_cache.get(key)
        if comb is None:
            comb = torch.combinations(torch.arange(n, device=device), r=4)
            self._comb_cache[key] = comb
        return comb

    def forward(self, student, teacher):
        n = student.size(0)
        if n < 4:
            return student.new_tensor(0.)

        comb = self._combinations(n, student.device)  # [num_sets, 4]

        def set_distance_sums(emb, chunk=2_000_000):
            # Matriz de distancia n x n (barata) em vez de materializar
            # [num_sets, 4, 4, dim] (~87 GB para n=128). Soma as 6 distancias
            # de pares por quadrupla, processando em blocos para limitar memoria.
            d = torch.cdist(emb, emb, p=2)  # [n, n]
            outs = []
            for s in range(0, comb.size(0), chunk):
                c = comb[s:s + chunk]
                outs.append(
                    d[c[:, 0], c[:, 1]] + d[c[:, 0], c[:, 2]] + d[c[:, 0], c[:, 3]]
                    + d[c[:, 1], c[:, 2]] + d[c[:, 1], c[:, 3]] + d[c[:, 2], c[:, 3]]
                )
            return torch.cat(outs)

        with torch.no_grad():
            t_sum = set_distance_sums(teacher)
            t_den = t_sum[t_sum > 0].mean()
            if torch.isnan(t_den) or t_den <= 0:
                t_den = t_sum.new_tensor(1.0)
            t_sum = t_sum / t_den

        s_sum = set_distance_sums(student)
        s_den = s_sum[s_sum > 0].mean()
        if torch.isnan(s_den) or s_den <= 0:
            s_den = s_sum.new_tensor(1.0)
        s_sum = s_sum / s_den

        loss = F.smooth_l1_loss(s_sum, t_sum, reduction='elementwise_mean')
        return loss
