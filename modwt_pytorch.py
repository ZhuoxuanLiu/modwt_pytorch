import numpy as np
import torch
import torch.nn as nn
import pywt


class WaveletTransform(nn.Module):
    def __init__(self, L, level, method) -> None:
        super().__init__()
        wavelet = pywt.Wavelet(method)
        self.level = level
        h = wavelet.dec_hi
        g = wavelet.dec_lo
        h_t = np.array(h) / np.sqrt(2)
        g_t = np.array(g) / np.sqrt(2)
        self.w_dec_filter = self.wavelet_dec_filter(h_t, L)
        self.v_dec_filter = self.wavelet_dec_filter(g_t, L)
        self.w_rec_filter = self.wavelet_rec_filter(h_t, L)
        self.v_rec_filter = self.wavelet_rec_filter(g_t, L)
        
    def wavelet_dec_filter(self, wavelet_vec, L):
        
        wavelet_len = len(wavelet_vec)   
        filter = torch.zeros(self.level, L, L)
        wl = torch.arange(wavelet_len)
        for j in range(self.level):
            for t in range(L):
                index = torch.remainder(t - 2 ** j * wl, L)
                hl = torch.zeros(L)
                for i, idx in enumerate(index):
                    hl[idx] = wavelet_vec[i]
                filter[j][t] = hl
        return filter   # (level, L, L)


    def wavelet_rec_filter(self, wavelet_vec, L):
        
        wavelet_len = len(wavelet_vec)   
        filter = torch.zeros(self.level, L, L)
        wl = torch.arange(wavelet_len)
        for j in range(self.level):
            for t in range(L):
                index = torch.remainder(t + 2 ** j * wl, L)
                hl = torch.zeros(L)
                for i, idx in enumerate(index):
                    hl[idx] = wavelet_vec[i]
                filter[j][t] = hl
        return filter   # (level, L, L)
    

    def modwt(self, x):
        '''
        x: (batch, length, D)
        filters: 'db1', 'db2', 'haar', ...
        '''
        B, L, D = x.shape
        x = x.permute(0, 2, 1)
        w_dec_filter = self.w_dec_filter.to(x)
        v_dec_filter = self.v_dec_filter.to(x)
        v_j = x
        v = []
        for j in range(self.level):
            v_j = torch.einsum('ml,bdl->bdm', v_dec_filter[j], v_j)
            v.append(v_j)
        v = torch.stack(v, dim=2)   # (B, D, level, L)
        v_prime = torch.cat([x.reshape(B, D, 1, L), v[..., :-1, :]], dim=2)  # (B, D, level, L)
        w = torch.einsum('jml,jbdl->bdjm', w_dec_filter, v_prime.permute(2, 0, 1, 3))
        wavecoeff = torch.cat([w, v[..., -1, :].reshape(B, D, 1, L)], dim=2)  # (B, D, level + 1, L)
        
        return wavecoeff.permute(0, 1, 3, 2)  # (B, D, L, level + 1)


    def imodwt(self, wave):
        '''
        wave: (batch, D, length, level + 1)
        '''
        wave = wave.permute(0, 1, 3, 2)
        w_rec_filter = self.w_rec_filter.to(wave)      # (level, L, L)
        v_rec_filter = self.v_rec_filter.to(wave)      # (level, L, L)
        w = wave[..., :-1, :]                               # (B, D, level, L)
        v_j = wave[..., -1, :]          # (B, D, L)
        scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[-1], v_j).unsqueeze(2)  # (B, D, 1, L)
        for j in range(self.level)[::-1]:
            detail_j = torch.einsum('ml,bdrl->bdrm', w_rec_filter[j], w[..., j, :].unsqueeze(2))
            scale_cat = torch.cat([detail_j, scale_j], dim=2)
            scale_j = torch.einsum('bdrl->bdl', scale_cat)
            if j > 0:
                scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[j - 1], scale_j).unsqueeze(2)  # (B, D, 1, L)
        
        return scale_j     # (B, D, L)
    
    
    def modwtmra(self, wave):
        ''' Multiresolution analysis based on MODWT'''
        '''
        wave: (batch, D, length, level + 1)
        '''
        wave = wave.permute(0, 1, 3, 2)
        w_rec_filter = self.w_rec_filter.to(wave)      # (level, L, L)
        v_rec_filter = self.v_rec_filter.to(wave)      # (level, L, L)
        w = wave[..., :-1, :]                               # (B, D, level, L)
        v_j = wave[..., -1, :]          # (B, D, L)
        scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[-1], v_j).unsqueeze(0)
        detail_j = torch.einsum('ml,nbdl->nbdm', w_rec_filter[-1], w[..., -1, :].unsqueeze(0))
        scale_j = torch.cat([detail_j, scale_j], dim=0)
        for j in range(self.level - 1)[::-1]:
            detail_j = torch.einsum('ml,nbdl->nbdm', w_rec_filter[j], w[..., j, :].unsqueeze(0))
            scale_j = torch.einsum('ml,nbdl->nbdm', v_rec_filter[j], scale_j)
            scale_j = torch.cat([detail_j, scale_j], dim=0)
        
        mra = scale_j.permute(1, 2, 0, 3)  # (B, D, level + 1, L)
        recon = torch.einsum('bdjl->bdl', mra)
        return mra.permute(0, 1, 3, 2), recon     # (B, D, L, level + 1)  (B, D, L)
