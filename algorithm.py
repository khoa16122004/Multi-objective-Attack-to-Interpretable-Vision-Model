# reference method:
import torch
import numpy as np
from explain_method import get_gradcam_map, simple_gradient_map, integrated_gradients
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# main attack
# class SpaEvoAtt():
#     def __init__(self,
#                 model,
#                 n = 4,  
#                 # 4, 16, 64, 256 only required for uni_rand: 4/(32*32) = 0.004 (CIFAR10)
#                 # 49, 196, 784, 3136 only required for uni_rand: 196/(224*224) = 0.004 (ImageNet)
#                 pop_size=10,
#                 cr=0.9,
#                 mu=0.01,
#                 seed = None,
#                 flag=True):

#         self.model = model
#         self.n_pix = n # if uni_rand is used
#         self.pop_size = pop_size
#         self.cr = cr
#         self.mu = mu
#         self.seed = seed
#         self.flag = flag

#     def convert1D_to_2D(self,idx,wi):
#         c1 = idx //wi
#         c2 = idx - c1 * wi
#         return c1, c2

#     def convert2D_to_1D(self,x,y,wi):
#         outp = x*wi + y
#         return outp

#     def masking(self,oimg,timg):
#         xo = torch.abs(oimg-timg)
#         d = torch.zeros(xo.shape[2],xo.shape[3]).bool().cuda()
#         for i in range (xo.shape[1]):
#             tmp = (xo[0,i]>0.).bool().cuda()
#             d = tmp | d # "or" => + ; |
        
#         wi = oimg.shape[2]
#         p = np.where(d.int().cpu().numpy() == 1) # oimg -> reference;'0' => "same as oimg" '1' => 'same as timg'
#         out = self.convert2D_to_1D(p[0],p[1],wi)

#         return out # output pixel coordinates have value same as 'timg'

#     def uni_rand(self,oimg,timg,olabel,tlabel):
    
#         if self.seed != None:
#             np.random.seed(self.seed)

#         terminate = False
#         nqry = 0
#         wi = oimg.shape[2]
#         he = oimg.shape[3]
        
#         fit = torch.zeros(self.pop_size) + np.inf
#         pop = []

#         p1 = np.zeros(wi*he).astype(int)
#         idxs = self.masking(oimg,timg) #[x for x in range(wi * he)]
#         p1[idxs] = 1
        
#         if p1.sum()<self.n_pix:
#             self.n_pix = p1.sum()        

#         for i in range(self.pop_size):
#             n = self.n_pix
#             cnt = 0
#             j = 0
#             while True:
#                 p = p1.copy()
#                 idx = np.random.choice(idxs, n, replace = False)
#                 p[idx] = 0
#                 nqry += 1
#                 fitness = self.feval(p,oimg,timg,olabel,tlabel)
                    
#                 if fitness < fit[i]:
#                     pop.append(p)
#                     fit[i] = fitness
#                     break
#                 elif (n>1):
#                     n -= 1
#                 elif (n == 1):
#                     while j < len(idxs):
#                         p[idxs[j]] = 0
#                         nqry += 1
#                         fitness = self.feval(p,oimg,timg,olabel,tlabel)

#                         if fitness < fit[i]:
#                             pop.append(p)
#                             fit[i] = fitness
#                             break
#                         else:
#                             j += 1
#                     break

#             if (j==len(idxs)-1):
#                 break
                
#         if len(pop)<self.pop_size:
#             for i in range(len(pop),self.pop_size):
#                 pop.append(p1)

#         return pop,nqry,fit

#     def recombine(self,p0,p1,p2):

#         cross_points = np.random.rand(len(p1)) < self.cr # uniform random
#         if not np.any(cross_points):
#             cross_points[np.random.randint(0, len(p1))] = True
#         trial = np.where(cross_points, p1, p2).astype(int)
#         trial = np.logical_and(p0,trial).astype(int) 
#         return trial

#     def mutate(self,p):

#         outp = p.copy()
#         if p.sum() != 0:
#             one = np.where(outp == 1)
#             n_px = int(len(one[0])*self.mu)
#             if n_px == 0:
#                 n_px = 1
#             idx = np.random.choice(one[0],n_px,replace=False)
#             outp[idx] = 0

#         return outp

#     def modify(self,pop,oimg,timg):
#         wi = oimg.shape[2]
#         img = timg.clone()
#         p = np.where(pop == 0)
#         c1,c2 = self.convert1D_to_2D(p[0],wi)
#         img[:,:,c1,c2] = oimg[:,:,c1,c2]
#         return img

#     def feval(self,pop,oimg,timg,olabel,tlabel):

#         xp = self.modify(pop,oimg,timg)
#         l2 = torch.norm(oimg - xp).cpu().numpy()
#         pred_label = self.model.predict_label(xp)

#         if self.flag == True:
#             if pred_label == tlabel:
#                 lc = 0
#             else:
#                 lc = np.inf
#         else:
#             if pred_label != olabel:
#                 lc = 0
#             else:
#                 lc = np.inf

#         outp = l2 + lc
#         return outp 


#     def selection(self,x1,f1,x2,f2):

#         xo = x1.copy()
#         fo = f1
#         if f2<f1:
#             fo = f2
#             xo = x2

#         return xo,fo

#     def evo_perturb(self,oimg,timg,olabel,tlabel,max_query=1000):

#         # 0. variable init
#         if self.seed != None:
#             np.random.seed(self.seed)

#         D = torch.zeros(max_query+500,dtype=int).cuda()
#         wi = oimg.shape[3]
#         he = oimg.shape[2]
#         n_dims = wi * he

#         # 1. population init
#         idxs = self.masking(oimg,timg) #[x for x in range(wi * he)]
#         if len(idxs)>1: # more than 1 diff pixel
#             pop, nqry,fitness = self.uni_rand(oimg,timg,olabel,tlabel)
            
#             if len(pop)>0:
#                 # 2. find the worst & best
#                 rank = np.argsort(fitness) 
#                 best_idx = rank[0].item()
#                 worst_idx = rank[-1].item()

#                 # ====== record ======
#                 D[:nqry] = l0(self.modify(pop[best_idx],oimg,timg),oimg)
#                 # ====================
                
#                 # 3. evolution
#                 while True:
#                     # a. Crossover (recombine)
#                     idxs = [idx for idx in range(self.pop_size) if idx != best_idx]
#                     id1, id2 = np.random.choice(idxs, 2, replace = False)
#                     offspring = self.recombine(pop[best_idx],pop[id1],pop[id2])

#                     # b. mutation (diversify)
#                     offspring = self.mutate(offspring)
                        
#                     # c. fitness evaluation
#                     fo = self.feval(offspring,oimg,timg,olabel,tlabel)
                        
#                     # d. select
#                     pop[worst_idx],fitness[worst_idx] = self.selection(pop[worst_idx],fitness[worst_idx],offspring,fo)
                        
#                     # e. update best and worst
#                     rank = np.argsort(fitness)
#                     best_idx = rank[0].item()
#                     worst_idx = rank[-1].item()

#                     # ====== record ======
#                     D[nqry] = l0(self.modify(pop[best_idx],oimg,timg),oimg)
#                     nqry += 1 
#                     # ====================
                    
#                     if nqry % 5000 == 0:
#                         print(pop[best_idx].sum().item(),nqry,self.model.predict_label(self.modify(pop[best_idx],oimg,timg)))
#                     if nqry > max_query:
#                         break
                
#                 # ====================

#                 adv = self.modify(pop[best_idx],oimg,timg)
#             else:
#                 adv = timg
#                 D[:nqry] = l0(self.modify(pop[best_idx],oimg,timg),oimg)#len(self.masking(oimg,timg))
#         else:
#             adv = timg
#             nqry = 1 # output purpose, not mean number of qry = 1
#             D[0] = 1
            
#         return adv, nqry, D[:nqry]



class NSGAII:
    def __init__(
        self,
        model,
        model_name,
        n=4,
        pop_size=20,
        cr=0.9,
        mu=0.01,
        topk=200,
        explain_method="gradcam",
        ig_steps=100,
        noise_std=1.0,
        noise_mode="generation",
        rgb_mutation_std=None,
        intersec_mode="reference",
        target_region=None,
        auto_region_percentile=30.0,
        target_objective="maximize_target_intersection",
        seed=None,
    ):
        self.model = model
        self.model_name = model_name
        self.n_pix = n
        self.pop_size = pop_size
        self.cr = cr
        self.mu = mu
        self.topk = topk
        self.explain_method = explain_method
        self.ig_steps = ig_steps
        # Keep parameter name for backward compatibility; it is now the RGB perturbation bound.
        self.noise_std = float(noise_std)
        self.noise_mode = str(noise_mode).lower()
        self.rgb_mutation_std = (
            float(rgb_mutation_std) if rgb_mutation_std is not None else max(1e-6, self.noise_std * 0.25)
        )
        self.intersec_mode = str(intersec_mode).lower()
        self.target_region = target_region
        self.auto_region_percentile = float(auto_region_percentile)
        self.target_objective = str(target_objective).lower()
        self.seed = seed

        if self.noise_mode not in ("fixed", "generation"):
            raise ValueError("noise_mode must be one of: fixed, generation")
        if self.intersec_mode not in ("reference", "target_region", "auto_low_region"):
            raise ValueError("intersec_mode must be one of: reference, target_region, auto_low_region")
        if self.target_objective not in ("maximize_target_intersection", "maximize_target_importance"):
            raise ValueError(
                "target_objective must be one of: maximize_target_intersection, maximize_target_importance"
            )
        if not (0.0 < self.auto_region_percentile < 100.0):
            raise ValueError("auto_region_percentile must be in (0, 100)")

        # Untargeted misclassification objective: minimize (logit_true - max_logit_other).
        # Negative margin means prediction is already not the original class.
        self.cls_margin_kappa = 0.0

    def _init_rgb_population(self, oimg, psize):
        c, h, w = oimg.shape[1], oimg.shape[2], oimg.shape[3]
        if self.noise_std <= 0:
            return torch.zeros((psize, c, h, w), device=oimg.device, dtype=oimg.dtype)
        return (torch.rand((psize, c, h, w), device=oimg.device, dtype=oimg.dtype) * 2.0 - 1.0) * self.noise_std

    def convert1D_to_2D(self, idx, wi):
        c1 = idx // wi
        c2 = idx - c1 * wi
        return c1, c2

    def convert2D_to_1D(self, x, y, wi):
        return x * wi + y

    def masking(self, oimg):
        wi = oimg.shape[2]
        he = oimg.shape[3]
        return torch.arange(wi * he, device=oimg.device, dtype=torch.long)

    def recombine(self, p1, p2, rgb1, rgb2):
        cross_points = torch.rand(p1.numel(), device=p1.device) < self.cr
        if not torch.any(cross_points):
            cross_points[torch.randint(0, p1.numel(), (1,), device=p1.device)] = True
        trial = torch.where(cross_points, p1, p2)
        child_mask = self._project_sparse(trial)

        _, h, w = rgb1.shape
        rgb_cross = cross_points.view(1, h, w)
        child_rgb = torch.where(rgb_cross, rgb1, rgb2)
        return child_mask, child_rgb

    def _project_sparse(self, p):
        outp = p.clone()
        target = int(getattr(self, "active_n_pix", self.n_pix))
        target = max(0, min(target, outp.numel()))

        one = torch.where(outp)[0]
        if one.numel() > target:
            n_off = int(one.numel() - target)
            off_idx = one[torch.randperm(one.numel(), device=outp.device)[:n_off]]
            outp[off_idx] = False
        elif one.numel() < target:
            zero = torch.where(~outp)[0]
            if zero.numel() > 0:
                on_cnt = min(target - int(one.numel()), int(zero.numel()))
                on_idx = zero[torch.randperm(zero.numel(), device=outp.device)[:on_cnt]]
                outp[on_idx] = True

        return outp

    def mutate(self, p, rgb):
        outp = p.clone()
        n_flip = int(max(1, round(outp.numel() * self.mu)))
        n_flip = min(n_flip, outp.numel())
        idx = torch.randperm(outp.numel(), device=outp.device)[:n_flip]
        outp[idx] = ~outp[idx]
        outp = self._project_sparse(outp)

        out_rgb = rgb.clone()
        _, h, w = out_rgb.shape
        n_rgb = int(max(1, round((h * w) * self.mu)))
        n_rgb = min(n_rgb, h * w)
        ridx = torch.randperm(h * w, device=out_rgb.device)[:n_rgb]
        rgb_flat = out_rgb.view(out_rgb.shape[0], -1)
        rgb_flat[:, ridx] += torch.randn(
            (out_rgb.shape[0], n_rgb),
            device=out_rgb.device,
            dtype=out_rgb.dtype,
        ) * self.rgb_mutation_std
        out_rgb = rgb_flat.view_as(out_rgb).clamp(-self.noise_std, self.noise_std)
        return outp, out_rgb

    def modify(self, pop, oimg, perturb_rgb):
        # Genome convention: 1 -> perturb this pixel, 0 -> keep original pixel.
        mask = pop.view(1, 1, oimg.shape[2], oimg.shape[3])
        perturbed = oimg + perturb_rgb.unsqueeze(0)
        img = torch.where(mask, perturbed, oimg)
        return img.clamp(oimg.min().item(), oimg.max().item())

    def _forward_logits(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _topk_idx(self, saliency_map):
        flat = saliency_map.reshape(-1)
        k = int(min(self.topk, flat.numel()))
        if k <= 0:
            raise ValueError("topk must be >= 1")
        return torch.topk(flat, k=k, largest=True).indices

    def _topk_intersection_ratio_batch(self, saliency_maps, ref_topk_idx):
        if saliency_maps.ndim == 2:
            saliency_maps = saliency_maps.unsqueeze(0)

        flat = saliency_maps.reshape(saliency_maps.size(0), -1)
        k = int(min(self.topk, flat.size(1)))
        if k <= 0:
            raise ValueError("topk must be >= 1")

        cur_idx = torch.topk(flat, k=k, dim=1, largest=True).indices
        ref_idx = ref_topk_idx.view(1, -1).to(cur_idx.device)
        matches = (cur_idx.unsqueeze(2) == ref_idx.unsqueeze(1)).any(dim=2)
        inter_counts = matches.sum(dim=1).to(torch.float32)
        return inter_counts / float(ref_idx.size(1))

    def _topk_in_region_ratio_batch(self, saliency_maps, region_mask):
        if saliency_maps.ndim == 2:
            saliency_maps = saliency_maps.unsqueeze(0)

        flat = saliency_maps.reshape(saliency_maps.size(0), -1)
        k = int(min(self.topk, flat.size(1)))
        if k <= 0:
            raise ValueError("topk must be >= 1")

        cur_idx = torch.topk(flat, k=k, dim=1, largest=True).indices
        region_flat = region_mask.reshape(-1).to(cur_idx.device)
        in_region = region_flat[cur_idx]
        return in_region.to(torch.float32).mean(dim=1)

    def _build_target_region_mask(self, oimg):
        h, w = oimg.shape[2], oimg.shape[3]
        region = self.target_region
        if region is None:
            raise ValueError("target_region must be provided when intersec_mode='target_region'")

        if torch.is_tensor(region):
            mask = region.to(device=oimg.device)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            if mask.ndim != 2:
                raise ValueError("target_region tensor must have shape [H, W] or [1, H, W]")
            if mask.shape[0] != h or mask.shape[1] != w:
                raise ValueError("target_region tensor shape must match input spatial size")
            return mask.bool()

        if isinstance(region, np.ndarray):
            if region.ndim != 2 or region.shape[0] != h or region.shape[1] != w:
                raise ValueError("target_region ndarray must have shape [H, W]")
            return torch.from_numpy(region).to(device=oimg.device).bool()

        if isinstance(region, (list, tuple)) and len(region) == 4:
            y1, y2, x1, x2 = [int(v) for v in region]
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            if y2 <= y1 or x2 <= x1:
                raise ValueError("target_region bbox must satisfy y2>y1 and x2>x1")
            mask = torch.zeros((h, w), device=oimg.device, dtype=torch.bool)
            mask[y1:y2, x1:x2] = True
            return mask

        raise ValueError("target_region must be one of: bbox tuple/list (y1,y2,x1,x2), [H,W] ndarray, or [H,W] tensor")

    def _build_auto_low_region_mask(self, ref_explain_map):
        ref = ref_explain_map
        if ref.ndim == 3:
            ref = ref[0]
        ref = ref.to(torch.float32)
        thr = torch.quantile(ref.reshape(-1), self.auto_region_percentile / 100.0)
        mask = ref <= thr
        # Ensure non-empty mask for numerical stability.
        if not torch.any(mask):
            flat = torch.zeros_like(ref, dtype=torch.bool).reshape(-1)
            flat[torch.argmin(ref.reshape(-1))] = True
            mask = flat.view_as(ref)
        return mask

    def _target_importance_objective_batch(self, saliency_maps, target_region_mask):
        if saliency_maps.ndim == 2:
            saliency_maps = saliency_maps.unsqueeze(0)

        flat = saliency_maps.reshape(saliency_maps.size(0), -1)
        region = target_region_mask.reshape(-1).to(flat.device)
        if not torch.any(region):
            return torch.zeros((flat.size(0),), device=flat.device, dtype=torch.float32)
        mean_in_region = flat[:, region].mean(dim=1)
        # Minimize negative importance to maximize importance in target region.
        return -mean_in_region.to(torch.float32)

    def _intersection_objective_batch(self, saliency_maps, ref_topk_idx, target_region_mask, target_topk_idx):
        if self.intersec_mode == "reference":
            return self._topk_intersection_ratio_batch(saliency_maps, ref_topk_idx)

        if self.target_objective == "maximize_target_importance":
            return self._target_importance_objective_batch(saliency_maps, target_region_mask)

        target_inter = self._topk_intersection_ratio_batch(saliency_maps, target_topk_idx)
        # Minimize negative intersection == maximize intersection with target map.
        return -target_inter

    def _get_explain_map(self, x, target_class):
        method = str(self.explain_method).lower()

        if isinstance(target_class, int):
            target_tensor = torch.full(
                (x.size(0),),
                int(target_class),
                device=x.device,
                dtype=torch.long,
            )
        elif isinstance(target_class, torch.Tensor):
            target_tensor = target_class.to(x.device).view(-1).long()
        else:
            target_tensor = torch.tensor(target_class, device=x.device, dtype=torch.long).view(-1)

        if method in ("gradcam", "cam"):
            explain_map, logits = get_gradcam_map(
                self.model,
                self.model_name,
                x,
                target_class=target_tensor,
            )
            explain_map_t = torch.as_tensor(explain_map, device=x.device, dtype=torch.float32)
            if explain_map_t.ndim == 2:
                explain_map_t = explain_map_t.unsqueeze(0)
            return explain_map_t, logits

        if method in ("simple_gradient", "simple", "sg"):
            explain_map, logits = simple_gradient_map(
                self.model,
                x,
                target_class=target_tensor,
            )
            return explain_map, logits

        if method in ("integrated_gradients", "ig"):
            explain_map, logits = integrated_gradients(
                self.model,
                x,
                target_class=target_tensor,
                steps=self.ig_steps,
            )
            return explain_map, logits

        raise ValueError(
            "Unknown explain_method. Use one of: gradcam, simple_gradient, integrated_gradients"
        )

    def _untargeted_margin_loss(self, logits, y_true):
        # logits: [B, C], y_true: [B]
        true_logit = logits.gather(1, y_true.view(-1, 1)).squeeze(1)
        other_logits = logits.clone()
        other_logits.scatter_(1, y_true.view(-1, 1), float("-inf"))
        max_other = other_logits.max(dim=1).values
        margin = true_logit - max_other
        if self.cls_margin_kappa > 0:
            return torch.clamp(margin, min=-self.cls_margin_kappa)
        return margin

    def calculating_crowding_distance(self, F):
        infinity = 1e14
        n_points = F.shape[0]
        n_obj = F.shape[1]

        if n_points <= 2:
            return np.full(n_points, infinity)

        I = np.argsort(F, axis=0, kind="mergesort")
        F_sorted = F[I, np.arange(n_obj)]

        dist = np.concatenate([F_sorted, np.full((1, n_obj), np.inf)]) - np.concatenate(
            [np.full((1, n_obj), -np.inf), F_sorted]
        )

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            if i - 1 >= 0:
                dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            if i + 1 < dist_to_next.shape[0]:
                dist_to_next[i, j] = dist_to_next[i + 1, j]

        norm = np.max(F_sorted, axis=0) - np.min(F_sorted, axis=0)
        norm[norm == 0] = np.nan

        dist_to_last = dist_to_last[:-1] / norm
        dist_to_next = dist_to_next[1:] / norm

        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        crowding[np.isinf(crowding)] = infinity
        return crowding

    def modify_population(self, population, oimg, perturb_rgb):
        if population.ndim != 2:
            raise ValueError("population must have shape [P, H*W]")
        psize = population.shape[0]
        h, w = oimg.shape[2], oimg.shape[3]
        if perturb_rgb.shape[0] != psize:
            raise ValueError("perturb_rgb must have shape [P, C, H, W]")
        mask = population.view(psize, 1, h, w)
        base = oimg.expand(psize, -1, -1, -1)
        batch = torch.where(mask, base + perturb_rgb, base)
        return batch.clamp(oimg.min().item(), oimg.max().item())

    def feval_population(
        self,
        population,
        oimg,
        perturb_rgb,
        olabel,
        ref_topk_idx,
        target_region_mask=None,
        target_topk_idx=None,
    ):
        xp = self.modify_population(population, oimg, perturb_rgb)

        explain_maps, logits = self._get_explain_map(xp, target_class=olabel)
        inter_ratio = self._intersection_objective_batch(
            explain_maps,
            ref_topk_idx,
            target_region_mask,
            target_topk_idx,
        )
        y = torch.full((logits.size(0),), int(olabel), device=logits.device, dtype=torch.long)
        cls_loss = self._untargeted_margin_loss(logits, y)

        return np.stack(
            [
                cls_loss.detach().cpu().numpy(),
                inter_ratio.detach().cpu().numpy(),
            ],
            axis=1,
        ).astype(np.float32)

    def init_population(self, oimg):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        wi = oimg.shape[2]
        he = oimg.shape[3]
        idxs = self.masking(oimg) # return array index của tất cả pixel
        self.active_n_pix = int(max(0, min(self.n_pix, len(idxs))))

        pop = torch.zeros((self.pop_size, wi * he), dtype=torch.bool, device=oimg.device)
        if self.active_n_pix > 0:
            for i in range(self.pop_size):
                sel = idxs[torch.randperm(idxs.numel(), device=oimg.device)[: self.active_n_pix]]
                pop[i, sel] = True
        rgb_pop = self._init_rgb_population(oimg, self.pop_size)
        return pop, rgb_pop

    def nsga2_selection(self, population, population_rgb, objectives, pop_size):
        fronts = NonDominatedSorting().do(objectives, only_non_dominated_front=False)
        survivors = []

        for front in fronts:
            front = np.asarray(front, dtype=np.int64)
            if front.size == 0:
                continue

            remaining = pop_size - len(survivors)
            if front.size <= remaining:
                survivors.extend(front.tolist())
                if len(survivors) >= pop_size:
                    break
                continue

            front_obj = objectives[front]
            crowding = self.calculating_crowding_distance(front_obj)
            order = np.argsort(-crowding)
            chosen = front[order[:remaining]].tolist()
            survivors.extend(chosen)
            if len(survivors) >= pop_size:
                break

        next_population = population[survivors]
        next_population_rgb = population_rgb[survivors]
        next_objectives = objectives[survivors]
        return next_population, next_population_rgb, next_objectives, survivors, fronts

    def solve(self, oimg, olabel, max_query=1000):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.history = []

        # Reference explain map from original image for objective-2.
        ref_explain_map, _ = self._get_explain_map(oimg, target_class=int(olabel))
        ref_topk_idx = self._topk_idx(ref_explain_map)
        target_region_mask = None
        target_topk_idx = None
        if self.intersec_mode == "target_region":
            target_region_mask = self._build_target_region_mask(oimg)
        elif self.intersec_mode == "auto_low_region":
            target_region_mask = self._build_auto_low_region_mask(ref_explain_map)

        if self.intersec_mode in ("target_region", "auto_low_region"):
            if target_region_mask is None:
                raise ValueError("target_region_mask must not be None for target-region modes")
            target_map = target_region_mask.to(ref_explain_map.device, dtype=torch.float32)
            target_topk_idx = self._topk_idx(target_map)

        population, population_rgb = self.init_population(oimg) # array of modified pixel indices + RGB perturbation tensors

        nqry = 0
        objectives = self.feval_population(
            population,
            oimg,
            population_rgb,
            olabel,
            ref_topk_idx,
            target_region_mask=target_region_mask,
            target_topk_idx=target_topk_idx,
        )
        best_ce = np.min(objectives[:, 0])
        best_inter = np.min(objectives[:, 1])


        nqry += len(population)
        if objectives.size > 0:
            r0 = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
            self.history.append(
                {
                    "nqry": int(nqry),
                    "best_ce": float(np.min(objectives[:, 0])),
                    "best_intersection": float(np.min(objectives[:, 1])),
                    "rank0_objectives": objectives[r0].copy(),
                    "rank0_rgb_abs_mean": float(population_rgb[r0].abs().mean().item()),
                }
            )

        while nqry < max_query:
            offspring = []
            offspring_rgb = []
            while len(offspring) < self.pop_size and nqry < max_query:
                idxs = torch.randperm(population.shape[0], device=population.device)[:2]
                child, child_rgb = self.recombine(
                    population[idxs[0]],
                    population[idxs[1]],
                    population_rgb[idxs[0]],
                    population_rgb[idxs[1]],
                )
                child, child_rgb = self.mutate(child, child_rgb)
                offspring.append(child)
                offspring_rgb.append(child_rgb)

            remaining = max_query - nqry
            offspring = offspring[:remaining]
            offspring_rgb = offspring_rgb[:remaining]
            if len(offspring) == 0:
                break
            offspring = torch.stack(offspring, dim=0)
            offspring_rgb = torch.stack(offspring_rgb, dim=0)
            offspring_obj = self.feval_population(
                offspring,
                oimg,
                offspring_rgb,
                olabel,
                ref_topk_idx,
                target_region_mask=target_region_mask,
                target_topk_idx=target_topk_idx,
            )
            nqry += len(offspring)

            combined_pop = torch.cat([population, offspring], dim=0)
            combined_pop_rgb = torch.cat([population_rgb, offspring_rgb], dim=0)
            if len(objectives) == 0:
                combined_obj = offspring_obj
            else:
                combined_obj = np.concatenate([objectives, offspring_obj], axis=0)



            # Keep this call as placeholder for your custom NSGA-II implementation.
            population, population_rgb, objectives, _, fronts = self.nsga2_selection(
                combined_pop,
                combined_pop_rgb,
                combined_obj,
                self.pop_size,
            )

            if objectives.size > 0:
                best_ce = np.min(objectives[:, 0])
                best_inter = np.min(objectives[:, 1])
                r0 = NonDominatedSorting().do(objectives, only_non_dominated_front=True)

                self.history.append(
                    {
                        "nqry": int(nqry),
                        "best_ce": float(best_ce),
                        "best_intersection": float(best_inter),
                        "rank0_objectives": objectives[r0].copy(),
                        "rank0_rgb_abs_mean": float(population_rgb[r0].abs().mean().item()),
                    }
                )
            
            # print("nqry: {}, best_ce: {:.4f}, best_intersection: {:.4f}".format(nqry, best_ce, best_inter))

        fronts = NonDominatedSorting().do(objectives, only_non_dominated_front=False)
        rank0 = fronts[0].tolist() if len(fronts) > 0 else []

        # Representative return: individual with minimum top-k intersection.
        if len(rank0) > 0:
            best_local = int(np.argmin(objectives[rank0, 1]))
            best_idx = int(rank0[best_local])
        else:
            best_idx = int(np.argmin(objectives[:, 1]))

        adv = self.modify(population[best_idx], oimg, population_rgb[best_idx])
        if len(rank0) > 0:
            rank0_batch = self.modify_population(population[rank0], oimg, population_rgb[rank0])
            rank0_advs = [rank0_batch[i : i + 1] for i in range(rank0_batch.shape[0])]
        else:
            rank0_advs = []
        return adv, rank0_advs, population, objectives, nqry, rank0


