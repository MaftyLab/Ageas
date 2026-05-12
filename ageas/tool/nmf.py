#!/usr/bin/env python3
"""
Experimental device-parallel NMF.

Hosts an in-progress :class:`NMFModel` that fans NMF runs out across
multiple GPUs (or CPU workers) using ``torch.multiprocessing``. Currently
exploratory: the consensus step is unimplemented, and the module is not
re-exported from :mod:`ageas.tool` by default.

author: jy
"""
import os
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torchnmf.nmf import NMF



class NMFModel(pl.LightningModule):
    """
    Device wise parallel NMF model with distributed models
    Not ideal for pytorch-lightning, but still GPU supported
    """
    def __init__(self,
                 ranks:int = [5,],
                 nmf_iter:int = 2,
                 master_seed:int = 42,
                 cache_dir:str = 'cache',
                 ):
        super().__init__()
        self.save_hyperparameters()
        seed_everything(master_seed)

        # Set units
        self.units = dict()
        for rank in self.hparams.ranks:
            self.units[rank] = {
                int(seed): {
                    'usage': None,
                    'spectra': None,
                    'mse': None,
                    'model_path': None,
                }
                for seed in torch.randint(0, 1000, (nmf_iter,))
            }
        
        # Make cache dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        
    def train(self,
              data,
              using_cuda:bool = False,
              devices:list = None,
              cpu_nparallel:int = 1,
              **kwargs):
        """
        Train the model in parallel
        """
        # Set device
        if using_cuda and torch.cuda.is_available():
            if devices is not None:
                ndevice = len(devices)
                n_launch = ndevice
            else:
                devices = range(torch.cuda.device_count())
                ndevice = torch.cuda.device_count()
                n_launch = ndevice
        else:
            ndevice = 0
            n_launch = cpu_nparallel

        # Get all units
        all_units = list()
        for rank in self.hparams.ranks:
            for seed in self.units[rank]:
                all_units.append((rank, seed))

        # Load to devices in parallel
        sbatch_size = len(all_units) // n_launch
        sortie_list = [
            all_units[
                i*sbatch_size: min((i+1)*sbatch_size, len(all_units))
            ]
            for i in range(n_launch)
        ]

        # Launch
        print(f'Launching {n_launch} parallel processes')
        print(f'Each process will handle {sbatch_size} tasks')
        print(f'Data shape: {data.shape}')
        manager = mp.Manager()
        return_dict = manager.dict()

        processes = list()
        for i,task in enumerate(sortie_list):
            device = f'cuda:{devices[i]}' if ndevice >= 1 else 'cpu'
            p = mp.Process(
                target = self._worker,
                args = (data, task, device, return_dict),
                kwargs = kwargs,
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        # Update units
        for rank, seed in return_dict:
            self.units[rank][seed] = return_dict[(rank, seed)]
            model = torch.load(
                self.units[rank][seed]['model_path'],
                map_location='cpu',
                weights_only=True
            )
            self.units[rank][seed]['spectra'] = model['H']
            self.units[rank][seed]['usage'] = model['W']

            # Clean the cache
            os.remove(self.units[rank][seed]['model_path'])
            self.units[rank][seed]['model_path'] = None

        return self.units

    def _worker(self,
                data,
                combos,
                device:str = 'cpu',
                return_dict:dict = None,
                **kwargs
                ):
        """
        Worker function for parallel training
        """
        # init 
        for rank, seed in combos:
            # Set model
            torch.manual_seed(seed)
            nmf = NMF(
                W = torch.randn(data.shape[1], rank, device=device).abs(),
                H = torch.randn(data.shape[0], rank, device=device).abs(),
                rank=rank
            ).to(device)

            # Train model
            nmf.fit(data.to(device), **kwargs)

            # Calculate MSE
            mse = F.mse_loss(data.to(device), nmf())

            # Save model
            model_path = os.path.join(self.cache_dir, f'{rank}_{seed}.pt')
            torch.save(nmf.state_dict(), model_path)

            # Save model
            return_dict[(rank, seed)] = {
                'usage': None,  
                'spectra': None,
                'model_path': model_path,
                'mse': mse.detach().to('cpu').float(),
            }
            torch.cuda.empty_cache()
        return

    def _refit(self,
               data,
               W,
               H,
               trainable_W:bool = True,
               trainable_H:bool = True,
               rank:int = None,
               device:str = 'cpu',
               **kwargs):
        """
        Could be used after obataining the consensus results
        """
        if W is not None and W is not None:
            assert W.shape[1] == H.shape[1]
        
        # Refit
        nmf = NMF(
            W = W.to(device) if W is not None else None,
            H = H.to(device) if H is not None else None,
            trainable_H = trainable_H,
            trainable_W = trainable_W,
            rank = rank
        ).to(device)
        nmf.fit(data.to(device), **kwargs)
        return NMF.W.detach().clone().cpu(), NMF.H.detach().clone().cpu()

    def consensus(self, units, **kwargs):
        """
        Consensus NMF
        Original one usage KMeans to find median spectra matrices
        and then refit the consensus spectra with the GEX for the consensus usage
        Maybe we can try something different here
        e.g. use wilcoxon to consensus spectra matrices

        TBDone
        """
        # Make consensus spectra first by clustering result spectra matrices

        # Then refit the consensus spectra with the GEX for the consensus usage
        return
    
    def clean_cache(self):
        """
        Clean the cache directory
        """
        for f in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, f))
        return


if __name__ == "__main__":
    V = torch.randn(1400, 200).abs()

    start_time = time.time()
    model = NMFModel(
        ranks = [5],
        nmf_iter = 2,
        master_seed = 42,
        cache_dir = '../cache'
    )
    model.train(
        V,
        # devices=[1, 4],
        # using_cuda=True,
        cpu_nparallel=2,
        max_iter=200
    )
    # print(model.units)
    print(f'Elapsed time: {time.time() - start_time:.2f} s')
    model.clean_cache()
    print('Done')