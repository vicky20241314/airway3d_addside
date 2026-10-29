"""Conservative background-facing cheek contours; exact projected mesh envelope."""
import numpy as np
import torch


def trusted_boundary(parsing, landmarks, stride=3):
    """Only skin against background, below nose base and above chin.

    Hair, ears, neck, clothing and image crop boundaries are never targets.
    This is a segmentation-supported silhouette hypothesis, not ground truth.
    """
    p = np.asarray(parsing)
    lm = np.asarray(landmarks)
    rows = []
    for y in range(max(2, int(lm[33, 1])), min(p.shape[0]-2, int(lm[8, 1])), stride):
        xs = np.flatnonzero(p[y] == 1)
        if len(xs) < 16:
            continue
        for side, x in ((-1, xs[0]), (1, xs[-1])):
            if x < 6 or x >= p.shape[1]-6:
                continue
            outside = p[y-1:y+2, x+1:x+6] if side == 1 else p[y-1:y+2, x-5:x]
            inside = p[y-1:y+2, x-3:x] if side == 1 else p[y-1:y+2, x+1:x+4]
            if np.mean(outside == 0) >= .8 and np.mean(inside == 1) >= .8:
                rows.append((y, x, side))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def envelope_pairs(uv, faces, rows):
    """Outermost triangle-edge intersection per scanline, no nearest vertex shortcut."""
    edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]]), axis=1), axis=0)
    a, b = uv[edges[:,0]], uv[edges[:,1]]
    selected, targets = [], []
    for y, x, side in rows:
        crossing = ((a[:,1] <= y) & (b[:,1] > y)) | ((b[:,1] <= y) & (a[:,1] > y))
        ids = np.flatnonzero(crossing)
        if not len(ids):
            continue
        t = (y-a[ids,1])/(b[ids,1]-a[ids,1])
        u = a[ids,0] + t*(b[ids,0]-a[ids,0])
        k = ids[np.argmin(u) if side < 0 else np.argmax(u)]
        selected.append(edges[k]); targets.append((y,x))
    return np.asarray(selected, dtype=int).reshape(-1,2), np.asarray(targets).reshape(-1,2)


class BoundaryConstraint:
    def __init__(self, parsing, landmarks, faces):
        self.rows = [trusted_boundary(p,l) for p,l in zip(parsing,landmarks)]
        if any(len(r) == 0 for r in self.rows):
            raise ValueError('No trustworthy background-facing cheek boundaries')
        self.faces = faces
        self.matches = []
        self.initialized = False
        self.scale = [max(8., np.linalg.norm(l[45]-l[36])) for l in landmarks]

    def update(self, projected):
        if not self.initialized:
            # Freeze accepted rows before optimization. Do not improve scores by
            # silently dropping hard rows later. Reject distant/shoulder edges.
            accepted = []
            for uv, rows, scale in zip(projected.detach().cpu().numpy(), self.rows, self.scale):
                e, q = envelope_pairs(uv, self.faces, rows)
                if len(e) != len(rows):
                    raise ValueError('Initial mesh does not cover boundary rows')
                a,b = uv[e[:,0]], uv[e[:,1]]
                slope = (b[:,0]-a[:,0])/(b[:,1]-a[:,1])
                pred = a[:,0]+(q[:,0]-a[:,1])*slope
                keep = (np.abs(pred-q[:,1]) < .2*scale) & (np.abs(slope)<2.)
                accepted.append(rows[keep])
            if any(len(r)==0 for r in accepted):
                raise ValueError('No boundary correspondences survive initial geometry gates')
            self.rows = accepted
            self.initialized = True
        self.matches = [envelope_pairs(u, self.faces, r) for u,r in zip(projected.detach().cpu().numpy(), self.rows)]

    def loss(self, projected):
        values = []
        for uv, (edges, target), scale in zip(projected, self.matches, self.scale):
            if not len(edges):
                continue
            e = torch.as_tensor(edges, device=uv.device)
            q = torch.as_tensor(target, device=uv.device, dtype=uv.dtype)
            a,b = uv[e[:,0]], uv[e[:,1]]
            dy = b[:,1]-a[:,1]
            dy = torch.where(dy.abs()<1e-5, torch.where(dy<0,-1e-5,1e-5),dy)
            t = ((q[:,0]-a[:,1])/dy).clamp(0,1)
            pred = a[:,0]+t*(b[:,0]-a[:,0])
            values.append(torch.nn.functional.huber_loss(pred/scale, q[:,1]/scale, delta=.02))
        return torch.stack(values).mean() if values else projected.sum()*0
