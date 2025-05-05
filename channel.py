import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
from pymanopt.manifolds import Product, Sphere
from pymanopt.function import pytorch
from pymanopt.manifolds import Euclidean

class GeometryBasedChannel(nn.Module):
    def __init__(self, Nt, Nr, NC, NR, device, subcarriers=1, wavelength=2.0, noise_power=1, siso=False):
        super().__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.NC = NC
        self.NR = NR
        self.K = subcarriers
        self.wavelength = wavelength
        self.noise_power = noise_power
        self.siso=siso

        self.total_paths = NC * NR
        self.d = 0.5 * wavelength

        # Internal channel state
        self.H = self._build_channel(device=device)  # to be updated with `update_channel`

        self.device = device

    def array_response(self, N, phi):
        n = torch.arange(N, dtype=torch.float32, device=phi.device).view(-1, 1)
        phase_shifts = 2 * math.pi * self.d * torch.sin(phi) / self.wavelength
        return (1 / math.sqrt(N)) * torch.exp(1j * n * phase_shifts)

    def _build_channel(self, device):
        if self.siso:
            return (torch.randn(1, device=device) + 1j * torch.randn(1, device=device)) / math.sqrt(2)
        
        else:
            if device is None:
                device = self.device if hasattr(self, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

            P = self.total_paths
            K = self.K
            H_all = []

            alpha = (torch.randn(P, device=device) + 1j * torch.randn(P, device=device)) / math.sqrt(2)
            psi = torch.rand(self.NC, device=device)
            psi_full = psi.repeat_interleave(self.NR)

            phi_t = (torch.rand(P, device=device) - 0.5) * math.pi
            phi_r = (torch.rand(P, device=device) - 0.5) * math.pi

            A_t = self.array_response(self.Nt, phi_t)
            A_r = self.array_response(self.Nr, phi_r)

            H_b = []
            for k in range(K):
                theta_k = 2 * math.pi * psi_full * k / K
                alpha_k = alpha * torch.exp(-1j * theta_k)
                H_k = A_r @ torch.diag(alpha_k) @ A_t.conj().transpose(0, 1)
                scaling = math.sqrt(self.Nt * self.Nr / P)
                H_b.append(scaling * H_k)
            
            return torch.stack(H_b, dim=0)  # [K, Nr, Nt]
    def get_channel(self):
        """
        Get the current channel realization.
        """
        return self.H

    def update_channel(self):
        """
        Regenerate a new channel realization and store it in self.H.
        """
        self.H = self._build_channel(device=self.device)  # [K, Nr, Nt]

    def forward(self, X):
        """
        Apply channel to transmitted signal.
        Args:
            X: [K, Nt] complex64 → transmitted signal per subcarrier
        Returns:
            Y: [K, Nr] → received signal
        """
        if self.siso:
            Y = self.H*X
            noise = (torch.randn_like(Y) + 1j * torch.randn_like(Y)) * math.sqrt(self.noise_power / 2)
            return self.H*X+noise
        else:
            K, Nt, _ = X.shape
            assert K == self.K and Nt == self.Nt

            Y = torch.matmul(self.H, X).squeeze(-1)  # [K, Nr]

            # Add complex AWGN
            noise = (torch.randn_like(Y) + 1j * torch.randn_like(Y)) * math.sqrt(self.noise_power / 2)
            return Y+noise  # [K, Nr]

    
class MAryModulation(nn.Module):
    def __init__(self, M=8):
        """
        M-ary modulation with hard forward / soft backward using fixed unit-circle constellation.
        Outputs complex numbers: x + j*y.
        """
        super().__init__()
        assert M & (M - 1) == 0, "M must be a power of 2"
        self.M = M
        self.bits_per_symbol = int(math.log2(M))
        self.constellation = self._generate_constellation(M)  # [M], complex dtype
        

    def _generate_constellation(self, M):
        """
        Generate M points evenly spaced on the unit circle in the complex plane.
        Returns complex-valued tensor of shape [M].
        """
        angles = 2 * torch.pi * torch.arange(M) / M
        constellation = torch.exp(1j * angles)
        return constellation  # [M]

    def modulate(self, x):
        """
        Args:
            x: Long tensor of shape [len], values in [0, 63]
        Returns:
            modulated: [len, 2], each entry is complex number (2 symbols per input)
        """
        if isinstance(x, list):
            L = len(x)
            B = x[0].shape[0]
            x = torch.cat(x, dim=0)  # [L * B, len, group]
        assert x.dtype == torch.long and x.max() < 64
        LB, length, group = x.shape # [B, len, group]
        flat_x = x.view(-1)  # [B * len * group]

        # Convert each value to 6-bit binary
        bits = ((flat_x.unsqueeze(-1) >> torch.arange(5, -1, -1, device=x.device)) & 1).float()  # [N, 6]
        bits_grouped = bits.view(-1, 2, 3)  # [N, 2, 3]
        weights = torch.tensor([4, 2, 1], device=x.device).view(1, 1, 3)
        symbol_indices = torch.sum(bits_grouped * weights, dim=-1).long()  # [N, 2]

        # Lookup in constellation
        constellation = self.constellation.to(x.device)  # [M]
        modulated = torch.stack([
            constellation[symbol_indices[:, 0]],
            constellation[symbol_indices[:, 1]]
        ], dim=-1)  # [N, 2]

        # Reshape back to [B, len, group, 2]
        return modulated.view(L, B, length, group, 2)  # complex tensor

    def demodulate(self, modulated):
        """
        Args:
            modulated: [len, 2] complex tensor (each value = x + j*y)
        Returns:
            recovered: [len] long tensor with values in [0, 63]
        """
        shape = modulated.shape[:-1]  # [B, len, group]
        flat_mod = modulated.view(-1, 2)  # [N, 2]
        constellation = self.constellation.to(modulated.device)  # [M]

        recovered_indices = []
        for i in range(2):
            symbols = flat_mod[:, i]  # [N]
            diff = symbols.unsqueeze(1) - constellation.unsqueeze(0)  # [N, M]
            dists = torch.abs(diff) ** 2
            indices = torch.argmin(dists, dim=-1)  # [N]
            recovered_indices.append(indices)

        # Merge back to 6-bit values
        high, low = recovered_indices
        recovered = (high << 3) + low  # [N]
        return recovered.view(*shape)  # [B, len, group]
    

def constant_modulus(matrix):
    # Given a complex matrix, project it to have unit modulus on every entry.
    return torch.exp(1j * torch.angle(matrix))



###############################
# Hybrid Precoders & Combiners
###############################
class HybridPrecoderModule(nn.Module):
    def __init__(self, Nt, NRF, Ns, subcarriers=1):
        """
        Nt: number of transmit antennas
        NRF: number of RF chains
        Ns: number of data streams per subcarrier
        """
        super(HybridPrecoderModule, self).__init__()
        self.Nt = Nt
        self.NRF = NRF
        self.Ns = Ns
        # Analog phases are common to all subcarriers.
        self.register_buffer("analog_phases", (torch.rand(Nt, NRF) * 2 * math.pi).to(torch.float32))
        # Digital precoder is subcarrier dependent: shape [K, NRF, Ns]
        angles = torch.rand(subcarriers, NRF, Ns) * 2 * math.pi
        self.register_buffer("digital_precoder", torch.exp(1j * angles).to(torch.complex64))

        self.register_buffer("beta", torch.rand(subcarriers, dtype=torch.float32))  # scaling factor for the digital precoder

    def get_analog_precoder(self):
        # Constant-modulus analog precoder: exp(j * phase)
        return torch.exp((1j * self.analog_phases).to(torch.complex64))  # shape: [Nt, NRF]

    def get_digital_precoder(self, k):
        # Return the digital precoder for subcarrier k.
        return self.digital_precoder[k].to(torch.complex64)

    def set_digital_precoder(self, k, new_value):
        # Update the digital precoder for subcarrier k.
        self.digital_precoder[k].copy_(new_value)

    def get_beta(self, k):
        # Return the digital precoder for subcarrier k.
        return self.beta[k].to(torch.float32)

    def set_beta(self, k, new_value):
        # Update the digital precoder for subcarrier k.
        self.beta[k].copy_(new_value)

    def get_V(self):
        """
        Compute the overall precoder for all subcarriers.
        V = Va @ Vd, where Va is common and Vd is subcarrier dependent.
        Returns V of shape [K, Nt, Ns]
        """
        VA = self.get_analog_precoder()                       # [Nt, NRF]
        VD = self.digital_precoder                         # [K, NRF, Ns]
        V = torch.matmul(VA, VD)                                # [K, Nt, Ns]
        return V

    def forward(self, s_t):
        # s_t: [K, Ns, 1]
        # Apply hybrid precoder: x_t = V[k] @ s_t[k]
        x_t = torch.matmul(self.get_V(), s_t).squeeze(-1)  # [K, Nt]
        return x_t

    def update_precoder_wmmse_scalar(self, H_k, Q_k, w_scalar, noise_var):
        """
        Closed‐form update for the overall precoder for subcarrier k.
        With fixed combiner Q_k and given weight (scalar) w_scalar, the update is:
        V_k* = (w * H_k^H Q_k Q_k^H H_k + noise_var I)^{-1} (w * H_k^H Q_k)
        """
        Va = self.get_analog_precoder()                       # [Nt, NRF]
        # 1. Form the effective channel: H_eff = H_k * Va, shape: [Nr, NRF]
        H_eff = Q_k.conj().T @ H_k @ Va  # shape: [Ns, NRF]

        # 2. Compute a scalar factor from the combiner:
        trace_Q = torch.trace(Q_k @ Q_k.conj().T).real  # scalar
        
        # 3. Compute the unscaled digital precoder V_u[k]:
        # Here, we build the matrix to invert. Note that the noise term is scaled by trace_Q/w_scalar.
        I_NRF = torch.eye(Va.shape[1], dtype=H_eff.dtype, device=H_eff.device)
        A = (H_eff.conj().T @ H_eff + (noise_var * trace_Q * Va.conj().T @ Va))
        B = H_eff.conj().T  # shape: [NRF, Ns]
        V_u = torch.linalg.solve(A, B)  # unscaled digital precoder [NRF, Ns]
        beta = 1.0
        # Compute the total power of the transmitted precoder: 
        # ||V_A * (V_u)||_F^2 = trace(Va * V_u * V_u^H * Va^H)
        prod = Va @ V_u
        power = torch.trace(prod @ prod.conj().T).real
        beta = 1.0 / power.sqrt()
        
        # 5. Final digital precoder update:
        Vd = beta * V_u

        return Vd, beta

    def optimize_analog_precoder_pymanopt(self, H, combiner_module, weight, noise_var, K):
        """
        Optimize the analog precoder phases using Pymanopt.
        Instead of using Circle, we use Sphere(1) (i.e. points on S^1 ⊂ ℝ²).
        """
        Nt, N_RF = self.analog_phases.shape  # self.analog_phases is stored as angles (in radians).

        # Manifold: optimize over Nt x N_RF real angles
        manifold = Euclidean(Nt, N_RF)
        
        # Ensure all relevant data is on CPU
        H = [h.cpu() for h in H]
        weight = weight.cpu()

        @pytorch(manifold)
        def cost_precoder(theta):
            # theta.requires_grad_(True)
            Va = (torch.cos(theta) + 1j * torch.sin(theta)).to(torch.complex64)

            J = 0.0
            for k in range(K):
                Vd_k = self.get_digital_precoder(k).detach().cpu()
                Qa = combiner_module.get_analog_combiner().detach().cpu()
                Qd_k = combiner_module.get_digital_combiner(k).detach().cpu()

                Q_k = Qa @ Qd_k
                J += weight[k].detach() * compute_J_k(H[k], Va, Q_k, noise_var)
                
            return J
        problem = Problem(manifold=manifold, cost=cost_precoder)
        solver = ConjugateGradient(max_iterations=200, verbosity=0)

        optimal_x = solver.run(problem).point
        new_angles = torch.tensor(optimal_x, dtype=self.analog_phases.dtype, device=self.analog_phases.device)
        self.analog_phases.copy_(new_angles)


class HybridCombinerModule(nn.Module):
    def __init__(self, Nr, NRF, Ns, subcarriers=1):
        """
        Nr: number of receive antennas
        NRF: number of RF chains
        Ns: number of data streams per subcarrier
        """
        super(HybridCombinerModule, self).__init__()
        self.Nr = Nr
        self.NRF = NRF
        self.Ns = Ns
        # Analog phases for combiner.
        self.register_buffer("analog_phases", (torch.rand(Nr, NRF) * 2 * math.pi).to(torch.float32))
        # Digital combiner is subcarrier dependent: shape [K, NRF, Ns]
        angles = torch.rand(subcarriers, NRF, Ns) * 2 * math.pi
        self.register_buffer("digital_combiner", torch.exp(1j * angles))
    
    def get_analog_combiner(self):
        return torch.exp((1j * self.analog_phases).to(torch.complex64))  # [Nr, NRF]
    
    def get_digital_combiner(self, k):
        return self.digital_combiner[k].to(torch.complex64)
    
    def set_digital_combiner(self, k, new_value):
        self.digital_combiner[k].copy_(new_value)

    def get_Q(self):
        """
        Compute the overall combiner for all subcarriers.
        Q = Qa @ Qd, where Qa is common and Qd is subcarrier dependent.
        Returns Q of shape [K, Nr, Ns]
        """
        QA = self.get_analog_combiner()                        # [Nr, NRF]
        QD = self.digital_combiner   # [K, NRF, Ns]
        Q = torch.matmul(QA, QD)                                # [K, Nr, Ns]
        
        return Q
    
    def forward(self, y_t):
        """
        Apply the combiner: for each subcarrier k, s_hat = Q[k]^H @ y[k]
        """
        QH = self.get_Q().conj().transpose(1, 2)  # [K, Ns, Nr]
        s_hat_t = torch.matmul(QH, y_t.unsqueeze(-1)).squeeze(-1)  # [K, Ns]
        return s_hat_t
    
    def update_combiner_wmmse_scalar(self, H_k, V_k, noise_var, w_scalar, beta_k):
        """
        Closed‐form update for the overall combiner for subcarrier k.
        With fixed precoder V_k, the update is:
        Q_k* = (H_k V_k V_k^H H_k^H + noise_var I)^{-1} H_k V_k
        """

        Qa = self.get_analog_combiner()                       # [Nt, NRF]

        H_eff = Qa.conj().T @ (H_k @ V_k)  # shape: [Ns, NRF]
        A = (H_eff @ H_eff.conj().T + (noise_var / (beta_k ** 2)) * (Qa.conj().T @ Qa))
        B = H_eff
        Qd_star = torch.linalg.solve(A, B)


        # Qd_star = Qd_star / torch.norm(Qd_star, p='fro') * math.sqrt(self.Ns)

        return Qd_star

    def optimize_analog_combiner_pymanopt(self, H, precoder_module, weight, noise_var, K):
        """
        Optimize the analog combiner phases using Pymanopt.
        We represent each element on the unit circle as a point on S¹ (a 2D unit vector).
        The manifold is defined as the product of Sphere(1) for each element in the analog combiner.
        """
        Nt, N_RF = self.analog_phases.shape  # self.analog_phases is stored as angles (in radians).

        # Manifold: optimize over Nt x N_RF real angles
        manifold = Euclidean(Nt, N_RF)
        
        # Ensure all relevant data is on CPU
        H = [h.cpu() for h in H]
        weight = weight.cpu()

        @pytorch(manifold)
        def cost_precoder(theta):
            # theta.requires_grad_(True)
            Qa = (torch.cos(theta) + 1j * torch.sin(theta)).to(torch.complex64)

            I = 0.0
            for k in range(K):
                Qd_k = self.get_digital_combiner(k).detach().cpu()
                Va = precoder_module.get_analog_precoder().detach().cpu()
                Vd_k = precoder_module.get_digital_precoder(k).detach().cpu()
                beta_k = precoder_module.get_beta(k).detach().cpu()

                V_k = Va @ Vd_k
                I += weight[k].detach() * compute_I_k(H[k], Qa, V_k, noise_var, beta_k)
                
            return I
        problem = Problem(manifold=manifold, cost=cost_precoder)
        solver = ConjugateGradient(max_iterations=200, verbosity=False)

        optimal_x = solver.run(problem).point
        new_angles = torch.tensor(optimal_x, dtype=self.analog_phases.dtype, device=self.analog_phases.device)
        self.analog_phases.copy_(new_angles)

#########################################
# Physical Layer Module
#########################################
class PhysicalLayerModule(nn.Module):
    def __init__(self, config, device=None):
        """
        config should contain:
          Nt, Nr, NRF, Ns, num_subcarriers, noise_power, and channel parameters.
        """
        super(PhysicalLayerModule, self).__init__()
        self.M = config.M  # Modulation order
        self.Nt = config.Nt
        self.Nr = config.Nr
        self.NRF = config.NRF
        self.Ns = config.Ns
        self.num_subcarriers = config.num_subcarriers
        self.snr = config.SNR  # scalar noise power
        self.siso = config.SISO

        self.power = 1.0

        self.noise_power = self.power/ (10 ** (self.snr / 10))
        
        # Channel parameters (for a geometry-based channel simulation)
        self.num_clusters = config.num_clusters
        self.num_rays = config.num_rays

        # Initialize modulation model
        self.mary_modulation = MAryModulation(M=self.M)
        
        # Initialize hybrid precoder and combiner modules
        self.precoder = HybridPrecoderModule(self.Nt, self.NRF, self.Ns, self.num_subcarriers)
        self.combiner = HybridCombinerModule(self.Nr, self.NRF, self.Ns, self.num_subcarriers)

        # Initialize geometry-based channel model
        self.channel = GeometryBasedChannel(self.Nt, self.Nr, self.num_clusters, self.num_rays, device=device, subcarriers=self.num_subcarriers, noise_power=self.noise_power, siso=self.siso)

    def reshape_modulated_symbols_to_grid(self, modulated_symbols):
        """
        Reshape modulated symbols [L, B, len, group, 2] → [T, K, Ns]
        where:
            T = B * len,
            K = L * group,
            Ns = 2
        """
        L, B, length, group, ns = modulated_symbols.shape
        assert ns == 2, "Last dimension should represent complex (real, imag)."
        x = modulated_symbols.permute(1, 2, 0, 3, 4)  # [B, len, L, group, 2]
        modulated_grid = x.reshape(B * length, L * group, 2)
        return modulated_grid  # [T, K, Ns]
    
    def reshape_grid_to_modulated_sequence(self, output_grid, shape):
        """
        Reshape output_grid [T, K, 2] → [L, B, len, group, 2] using shape tuple.
        """
        L, B, length, group, _ = shape
        T, K, Ns = output_grid.shape
        assert T == B * length
        assert K == L * group
        assert Ns == 2
        x = output_grid.reshape(B, length, L, group, 2)
        output_sequence = x.permute(2, 0, 1, 3, 4).contiguous()
        return output_sequence

    def forward(self, symbols, weight=None):
        """
        Forward pass through hybrid precoding, MIMO-OFDM channel, and combining.
        """
        if self.siso:
            self.channel.update_channel()
            H = self.channel.get_channel()
            modulated_symbols = self.mary_modulation.modulate(symbols) # [L, B, len, group, 2] complex
            transmitted_symbols = modulated_symbols.flatten()

            received_symbols = self.channel(transmitted_symbols)/ H
            received_symbols = received_symbols.reshape(modulated_symbols.shape)
            demodulate_symbol = self.mary_modulation.demodulate(received_symbols)

        else:
            modulated_symbols = self.mary_modulation.modulate(symbols)  # [L, B, len, group, 2] complex
            modulated_grid = self.reshape_modulated_symbols_to_grid(modulated_symbols)  # [T, K, Ns]
            T, K, Ns = modulated_grid.shape
            assert K == self.num_subcarriers
            output_sequence = []
            self.channel.update_channel()      # update channel for current time step
            H = self.channel.get_channel()     # [K, Nr, Nt]
            mean_weight = weight.mean(dim=0)
            if is_main_process() :
                print(f"mean_weight: {mean_weight}")
            # Perform WMMSE optimization (updates precoder and combiner in-place)
            self.wmmse_hybrid_beamforming(H, mean_weight, noise_var=self.noise_power)
            for t in range(T): 
                s_t = modulated_grid[t]            # [K, Ns]
                # weight_t = weight[t]
                s_t = s_t.unsqueeze(-1)            # [K, Ns, 1]
                   

                x_t = self.precoder(s_t)           # [K, Nt]
                y_t = self.channel(x_t.unsqueeze(-1))  # [K, Nr]
                s_hat_t = self.combiner(y_t).squeeze(0)  # [K, Ns]
                beta_all = torch.stack([self.precoder.get_beta(k) for k in range(K)])  # [K]
                beta_all = beta_all.view(K, 1)

                # Undo beta scaling at receiver for fair comparison
                s_hat_rescaled = s_hat_t / beta_all  # [K, Ns]
                s_true = s_t.squeeze(-1)             # [K, Ns]

                # mse_direct = torch.mean(torch.abs(s_true - s_hat_rescaled) ** 2)
                # print("s", s_true[0], "s_hat", s_hat_rescaled[0])
                # print("Empirical MSE (post β rescaling):", mse_direct.item())
                output_sequence.append(s_hat_rescaled)

            output_grid = torch.stack(output_sequence, dim=0)  # [T, K, Ns]
            output_sequence = self.reshape_grid_to_modulated_sequence(output_grid, modulated_symbols.shape)  # [L, B, len, group, 2]
            demodulate_symbol = self.mary_modulation.demodulate(output_sequence)
        return demodulate_symbol  # [T, K, Ns]
        # return output_grid
        
    def wmmse_hybrid_beamforming(self, H, weight, noise_var=1e-3, num_iters=3):
        """
        Perform the WMMSE optimization in two phases:
        Phase 1: Optimize precoders (digital & analog) with combiners fixed.
        Phase 2: Optimize combiners (digital & analog) with precoders fixed.
        The provided weight vector (of shape [K]) remains fixed.
        The analog updates are performed via Pymanopt.
        """
        K = H.shape[0]
        
        for it in range(num_iters):
            # --- Phase 1: Update Precoder (Digital + Analog) with combiner fixed ---
            # Update analog precoder using Pymanopt.
            self.precoder.optimize_analog_precoder_pymanopt(H, self.combiner, weight, noise_var, K)

            # Update digital precoder again using the analog precoder.
            for k in range(K):
                Va = self.precoder.get_analog_precoder()         # [Nt, NRF]
                Vd = self.precoder.get_digital_precoder(k)          # [NRF, Ns]
                V_k = Va @ Vd                                       # [Nt, Ns]
                Qa = self.combiner.get_analog_combiner()            # [Nr, NRF]
                Qd = self.combiner.get_digital_combiner(k)          # [NRF, Ns]
                Q_k = Qa @ Qd                                       # [Nr, Ns]
                # Compute the WMMSE update for the digital precoder.
                Vd_k_star, beta = self.precoder.update_precoder_wmmse_scalar(H[k], Q_k, weight[k], noise_var)
                self.precoder.set_digital_precoder(k, Vd_k_star)
                self.precoder.set_beta(k, beta)



            # --- Phase 2: Update Combiner (Digital + Analog) with precoder fixed ---
            # Update analog combiner using Pymanopt.
            self.combiner.optimize_analog_combiner_pymanopt(H, self.precoder, weight, noise_var, K)

            for k in range(K):
                Va = self.precoder.get_analog_precoder()         # [Nt, NRF]
                Vd = self.precoder.get_digital_precoder(k)          # [NRF, Ns]
                V_k = Va @ Vd                                       # [Nt, Ns]
                beta = self.precoder.get_beta(k)
                Qd_k_star = self.combiner.update_combiner_wmmse_scalar(H[k], V_k, noise_var, weight[k], beta)
                self.combiner.set_digital_combiner(k, Qd_k_star)
                
            # Optionally, compute the total objective.
            total_obj = 0.0
            for k in range(K):
                Va = self.precoder.get_analog_precoder()
                Vd = self.precoder.get_digital_precoder(k)
                V_k = Va @ Vd
                beta = self.precoder.get_beta(k)
                Qa = self.combiner.get_analog_combiner()
                Qd = self.combiner.get_digital_combiner(k)
                Q_k = Qa @ Qd     
                MSE_k = compute_weighted_mse(H[k], Va, Vd, Qa, Qd, beta, noise_var)
                total_obj += weight[k] * MSE_k
                if is_main_process() and k==0:
                    print(f"[iter {it+1}] Subcarrier {k}: trace(MSE_k) = {MSE_k:.4f}, ")
            if is_main_process():
                print(f"[Iter {it+1}/{num_iters}] Total Weighted WMMSE Objective = {total_obj.item():.4f}")
            


###############################
# Helper Function
###############################
def compute_J_k(H_k, Va, Q_k, noise_var):
    """
    Implements Equation (31):
    J_k(Va) = tr( (I + 1 / (σ² * tr(Q^H Q)) * H^H Va (Va^H Va)^(-1) Va^H H )^(-1) )
    """
    Ns = Q_k.shape[1]  # output dimension
    I_Ns = torch.eye(Ns, dtype=Va.dtype, device=Va.device)

    QH_Q = Q_k.conj().T @ Q_k  # [Ns x Ns]
    tr_QH_Q = torch.trace(QH_Q).real

    # Compute the key intermediate product:
    VaHV = Va.conj().T @ Va  # [NRF x NRF]
    VaHV_inv = torch.linalg.inv(VaHV)

    H_effk = (H_k.conj().T @Q_k)  # [Nt x Nr]

    middle = H_effk.conj().T @ Va @ VaHV_inv @ Va.conj().T @ H_effk  # [Nt x Nt]

    scale = 1.0 / (noise_var * tr_QH_Q)
    A = I_Ns + scale * middle

    Jk = torch.trace(torch.linalg.inv(A)).real
    return Jk


def compute_I_k(H_k, Qa, V_k, noise_var, beta):
    """
    Implements Equation (31):
    J_k(Va) = tr( (I + 1 / (σ² * tr(Q^H Q)) * H^H Va (Va^H Va)^(-1) Va^H H )^(-1) )
    """
    Ns = V_k.shape[1]  # output dimension
    I_Ns = torch.eye(Ns, dtype=Qa.dtype, device=Qa.device)

    # Compute the key intermediate product:
    QaHQ = Qa.conj().T @ Qa  # [NRF x NRF]
    QaHQ_inv = torch.linalg.inv(QaHQ)

    H_effk = (H_k@V_k)  # [Nt x Nr]

    middle = H_effk.conj().T @ Qa @ QaHQ_inv @ Qa.conj().T @ H_effk  # [Nt x Nt]

    scale = (beta ** 2) / noise_var
    A = I_Ns + scale * middle

    Ik = torch.trace(torch.linalg.inv(A)).real
    return Ik

def compute_weighted_mse(H_k, Va, Vd_k, Qa, Qd_k, beta_k, noise_var):
    """
    Compute the weighted MSE for subcarrier k using Equation (4).

    H_k: [Nr, Nt]
    Va: [Nt, NRF]  (analog precoder)
    Vd_k: [NRF, Ns] (digital precoder)
    Qa: [Nr, NRF]  (analog combiner)
    Qd_k: [NRF, Ns] (digital combiner)
    beta_k: scalar
    noise_var: scalar
    """

    V_k = Va @ Vd_k         # [Nt, Ns]
    Q_k = Qa @ Qd_k         # [Nr, Ns]
    Q_H = Q_k.conj().T      # [Ns, Nr]

    HV = H_k @ V_k          # [Nr, Ns]
    QHHV = Q_H @ HV         # [Ns, Ns]
    QHQ = Q_H @ Q_k         # [Ns, Ns]

    term1 = (1 / beta_k**2) * Q_H @ HV @ HV.conj().T @ Q_k     # tr[ β⁻² Wᴴ H V Vᴴ Hᴴ W ]
    term2 = (1 / beta_k) * QHHV                                # tr[ -β⁻¹ Wᴴ H V ]
    term3 = (1 / beta_k) * QHHV.conj().T                       # tr[ -β⁻¹ Vᴴ Hᴴ W ]
    term4 = (noise_var / beta_k**2) * QHQ                      # tr[ σ² β⁻² Wᴴ W ]
    Ns = Vd_k.shape[1]
    I = torch.eye(Ns, dtype=Va.dtype, device=Va.device)

    mse_matrix = term1 - term2 - term3 + term4 + I
    mse_scalar = torch.trace(mse_matrix).real
    # if is_main_process():
    #     print(f"MSE terms: {torch.trace(term1).real.item():.3f}, {torch.trace(term2).real.item():.3f}, {torch.trace(term3).real.item():.3f}, {torch.trace(term4).real.item():.3f}")
    return mse_scalar

def is_main_process():
    """Check if this is the main process in DDP training or single-process training."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
