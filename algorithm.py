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
        self.seed = seed

        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")

    def convert1D_to_2D(self, idx, wi):
        c1 = idx // wi
        c2 = idx - c1 * wi
        return c1, c2

    def convert2D_to_1D(self, x, y, wi):
        return x * wi + y

    def masking(self, oimg, timg):
        xo = torch.abs(oimg - timg)
        d = torch.zeros(xo.shape[2], xo.shape[3], dtype=torch.bool, device=xo.device)
        for i in range(xo.shape[1]):
            d = (xo[0, i] > 0.0) | d

        wi = oimg.shape[2]
        p = np.where(d.int().detach().cpu().numpy() == 1)
        return self.convert2D_to_1D(p[0], p[1], wi)

    def recombine(self, p0, p1, p2):
        cross_points = np.random.rand(len(p1)) < self.cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, len(p1))] = True
        trial = np.where(cross_points, p1, p2).astype(int)
        trial = np.logical_and(p0, trial).astype(int)
        return trial

    def mutate(self, p):
        outp = p.copy()
        if p.sum() != 0:
            one = np.where(outp == 1)[0]
            n_px = int(len(one) * self.mu)
            if n_px == 0:
                n_px = 1
            n_px = min(n_px, len(one))
            idx = np.random.choice(one, n_px, replace=False)
            outp[idx] = 0
        return outp

    def modify(self, pop, oimg, timg):
        wi = oimg.shape[2]
        img = timg.clone()
        p = np.where(pop == 0)
        c1, c2 = self.convert1D_to_2D(p[0], wi)
        img[:, :, c1, c2] = oimg[:, :, c1, c2]
        return img

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

    def modify_population(self, population, oimg, timg):
        batch = timg.repeat(len(population), 1, 1, 1)
        wi = oimg.shape[2]
        src = oimg[0]

        for i, pop in enumerate(population):
            p = np.where(pop == 0)
            c1, c2 = self.convert1D_to_2D(p[0], wi)
            batch[i, :, c1, c2] = src[:, c1, c2]

        return batch

    def feval_population(self, population, oimg, timg, olabel, ref_topk_idx):
        xp = self.modify_population(population, oimg, timg)
        explain_maps, logits = self._get_explain_map(xp, target_class=y)
        y = torch.full((logits.size(0),), int(olabel), device=logits.device, dtype=torch.long)
        ce = self.ce_loss(logits, y)
        inter_ratio = self._topk_intersection_ratio_batch(explain_maps, ref_topk_idx)

        return np.stack(
            [
                ce.detach().cpu().numpy(),
                inter_ratio.detach().cpu().numpy(),
            ],
            axis=1,
        ).astype(np.float32)

    def init_population(self, oimg, timg):
        if self.seed is not None:
            np.random.seed(self.seed)

        wi = oimg.shape[2]
        he = oimg.shape[3]
        base = np.zeros(wi * he).astype(int)
        idxs = self.masking(oimg, timg)
        base[idxs] = 1

        if base.sum() < self.n_pix:
            self.n_pix = int(base.sum())

        pop = []
        for _ in range(self.pop_size):
            p = base.copy()
            if len(idxs) > 0 and self.n_pix > 0:
                n = min(self.n_pix, len(idxs))
                rm_idx = np.random.choice(idxs, n, replace=False)
                p[rm_idx] = 0
            pop.append(p)
        return pop

    def nsga2_selection(self, population, objectives, pop_size):
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

        next_population = [population[i] for i in survivors]
        next_objectives = objectives[survivors]
        return next_population, next_objectives, survivors, fronts

    def solve(self, oimg, timg, olabel, max_query=1000):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.history = []

        # Reference explain map from original image for objective-2.
        ref_explain_map, _ = self._get_explain_map(oimg, target_class=int(olabel))
        ref_topk_idx = self._topk_idx(ref_explain_map)

        population = self.init_population(oimg, timg)

        nqry = 0
        init_count = min(len(population), max_query - nqry)
        population = population[:init_count]
        objectives = self.feval_population(population, oimg, timg, olabel, ref_topk_idx)
        nqry += len(population)
        if objectives.size > 0:
            self.history.append(
                {
                    "nqry": int(nqry),
                    "best_ce": float(np.min(objectives[:, 0])),
                    "best_intersection": float(np.min(objectives[:, 1])),
                }
            )

        while nqry < max_query:
            offspring = []
            while len(offspring) < self.pop_size and nqry < max_query:
                idxs = np.random.choice(len(population), 3, replace=False)
                child = self.recombine(population[idxs[0]], population[idxs[1]], population[idxs[2]])
                child = self.mutate(child)
                offspring.append(child)

            remaining = max_query - nqry
            offspring = offspring[:remaining]
            if len(offspring) == 0:
                break
            offspring_obj = self.feval_population(offspring, oimg, timg, olabel, ref_topk_idx)
            nqry += len(offspring)

            combined_pop = population + offspring
            if len(objectives) == 0:
                combined_obj = offspring_obj
            else:
                combined_obj = np.concatenate([objectives, offspring_obj], axis=0)

            # Keep this call as placeholder for your custom NSGA-II implementation.
            population, objectives, _, fronts = self.nsga2_selection(
                combined_pop,
                combined_obj,
                self.pop_size,
            )

            if objectives.size > 0:
                self.history.append(
                    {
                        "nqry": int(nqry),
                        "best_ce": float(np.min(objectives[:, 0])),
                        "best_intersection": float(np.min(objectives[:, 1])),
                    }
                )

        fronts = NonDominatedSorting().do(objectives, only_non_dominated_front=False)
        rank0 = fronts[0].tolist() if len(fronts) > 0 else []

        # Representative return: the one with minimum CE in rank-0 (or entire pop if needed).
        if len(rank0) > 0:
            best_local = int(np.argmin(objectives[rank0, 0]))
            best_idx = int(rank0[best_local])
        else:
            best_idx = int(np.argmin(objectives[:, 0]))

        adv = self.modify(population[best_idx], oimg, timg)
        return adv, population, objectives, nqry, rank0


