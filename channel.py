import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

class GeometryBasedChannel(nn.Module):
    def __init__(self, Nt, Nr, NC, NR, device, subcarriers=1, wavelength=2.0, noise_power=1e-3):
        super().__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.NC = NC
        self.NR = NR
        self.K = subcarriers
        self.wavelength = wavelength
        self.noise_power = noise_power

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
        H_all.append(torch.stack(H_b, dim=0))  # [K, Nr, Nt]

        return torch.stack(H_all, dim=0)  # [K, Nr, Nt]
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
        K, Nt, _ = X.shape
        assert K == self.K and Nt == self.Nt

        Y = torch.matmul(self.H, X).squeeze(-1)  # [K, Nr]

        # Add complex AWGN
        noise = (torch.randn_like(Y) + 1j * torch.randn_like(Y)) * math.sqrt(self.noise_power / 2)
        return Y+noise   # [K, Nr]

    
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


##########################################
# 2. WMMSE-Based Hybrid Beamforming Optimizer
##########################################
def update_digital_precoder(H, VA, noise_power):
    """
    Given channel H [Nr, Nt] and current analog precoder VA [Nt, NRF],
    update digital precoder VD by a closed‐form MMSE update.
    Here we use a simplified MMSE formulation:
    
        VD = (VA^H H^H H VA + σ^2 I)^{-1} (VA^H H^H)
    
    so that the overall precoder is V = VA * VD.
    """
    NRF = VA.shape[1]
    I = torch.eye(NRF, dtype=VA.dtype, device=VA.device)
    A = torch.matmul(VA.conj().t(), torch.matmul(H.conj().t(), H) @ VA)
    VD = torch.linalg.solve(A + noise_power * I, torch.matmul(VA.conj().t(), H.conj().t()))
    return VD

def update_digital_combiner(H, QA, noise_power):
    """
    Similarly, given channel H and analog combiner QA [Nr, NRF],
    update digital combiner QD:
    
        QD = (QA^H H H^H QA + σ^2 I)^{-1} (QA^H H)
    """
    NRF = QA.shape[1]
    I = torch.eye(NRF, dtype=QA.dtype, device=QA.device)
    A = torch.matmul(QA.conj().t(), H @ H.conj().t() @ QA)
    QD = torch.linalg.solve(A + noise_power * I, torch.matmul(QA.conj().t(), H))
    return QD

def manifold_gradient_step(phase_param, objective, lr):
    """
    Perform one Riemannian gradient descent step on the analog phase parameters.
    phase_param: current phase parameters (real tensor)
    objective: scalar objective from which we compute gradient w.r.t. phase_param.
    lr: step size.
    Returns: updated phase parameters.
    """
    # Compute gradient with respect to phase_param.
    grad = torch.autograd.grad(objective, phase_param, retain_graph=True)[0]
    # Simple gradient descent step
    new_phase = phase_param - lr * grad
    return new_phase

def wmmse_hybrid_beamforming_opt(H, precoder_module, combiner_module, noise_power, weight=1.0, num_iter=10, analog_lr=0.01):
    """
    Alternating minimization to optimize the hybrid precoder and combiner
    for a given channel H.
    
    H: channel matrix, shape [Nr, Nt] (complex)
    precoder_module: an instance of HybridPrecoderModule
    combiner_module: an instance of HybridCombinerModule
    noise_power: scalar noise power
    weight: weight parameter from WMMSE formulation (if needed)
    num_iter: number of alternating iterations
    analog_lr: step size for manifold (analog phase) updates.
    
    Returns: optimized overall precoder V and combiner Q.
    """
    # We assume that the modules’ analog parameters are our free variables.
    # We detach the digital precoders from previous training (they are updated in closed-form).
    for it in range(num_iter):
        # Digital update:
        VA = precoder_module.get_analog_precoder()  # [Nt, NRF]
        VD = update_digital_precoder(H, VA, noise_power)  # [NRF, Ns]
        precoder_module.digital_precoder.data.copy_(VD)
        V = torch.matmul(VA, VD)  # Overall precoder
        
        QA = combiner_module.get_analog_combiner()  # [Nr, NRF]
        QD = update_digital_combiner(H, QA, noise_power)  # [NRF, Ns]
        combiner_module.digital_combiner.data.copy_(QD)
        Q = torch.matmul(QA, QD)
        
        # Compute the effective channel: H_eff = Q^H H V
        H_eff = torch.matmul(Q.conj().t(), torch.matmul(H, V))
        # A typical weighted MMSE objective (simplified):
        # J = tr((I + (1/noise_power) * H_eff^H H_eff)^{-1})
        I = torch.eye(H_eff.shape[0], dtype=H_eff.dtype, device=H_eff.device)
        J = torch.trace(torch.linalg.inv(I + (1/noise_power) * torch.matmul(H_eff.conj().t(), H_eff)))
        
        # For analog update, compute gradient with respect to the phase parameters.
        # We update the precoder and combiner analog phases separately.
        # Make sure the phase parameters require grad.
        if not precoder_module.analog_phases.requires_grad:
            precoder_module.analog_phases.requires_grad_(True)
        if not combiner_module.analog_phases.requires_grad:
            combiner_module.analog_phases.requires_grad_(True)
        
        # Compute gradients (using autograd) for the analog phase parameters.
        # Here we treat J as the objective to minimize.
        J.backward(retain_graph=True)
        
        # Update analog phases for precoder
        with torch.no_grad():
            new_precoder_phase = manifold_gradient_step(precoder_module.analog_phases, J, analog_lr)
            precoder_module.analog_phases.copy_(new_precoder_phase)
            new_combiner_phase = manifold_gradient_step(combiner_module.analog_phases, J, analog_lr)
            combiner_module.analog_phases.copy_(new_combiner_phase)
        
        # Clear gradients for next iteration.
        precoder_module.analog_phases.grad = None
        combiner_module.analog_phases.grad = None
        
        # Optionally print or log the objective value.
        print(f"Iteration {it+1}/{num_iter}, WMMSE Objective J = {J.item():.4f}")
    
    # After optimization, return the overall beamforming matrices.
    VA_opt = constant_modulus(torch.exp(1j * precoder_module.analog_phases))
    VD_opt = precoder_module.digital_precoder
    V_opt = torch.matmul(VA_opt, VD_opt)
    
    QA_opt = constant_modulus(torch.exp(1j * combiner_module.analog_phases))
    QD_opt = combiner_module.digital_combiner
    Q_opt = torch.matmul(QA_opt, QD_opt)
    
    return V_opt, Q_opt

#############################
# Hybrid Precoders & Combiners
#############################

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
        self.register_buffer("analog_phases", (torch.rand(Nt, NRF) * 2 * math.pi).to(torch.float32))
        angles = torch.rand(subcarriers, NRF, Ns) * 2 * math.pi
        self.register_buffer("digital_precoder", torch.exp(1j * angles))

    
    def get_analog_precoder(self):
        # Constant-modulus analog precoder: exp(j*phase)
        return torch.exp(1j * self.analog_phases)  # shape: [Nt, NRF]
    
    def get_V(self):
        VA = self.get_analog_precoder().to(torch.complex64)                    # [Nt, NRF]
        VD = self.digital_precoder.to(torch.complex64)                          # [K, NRF, Ns]
        V = torch.matmul(VA, VD)                                                # [K, Nt, Ns]

        return V

    def forward(self, s_t):
        
        # Apply hybrid precoder: X[k] = V[k] @ s[k]
        x_t = torch.matmul(self.get_V(), s_t).squeeze(-1)             # [K, Nt]

        return x_t

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
        self.register_buffer("analog_phases", (torch.rand(Nr, NRF) * 2 * math.pi).to(torch.float32))
        angles = torch.rand(subcarriers, NRF, Ns) * 2 * math.pi
        self.register_buffer("digital_combiner", torch.exp(1j * angles))
    
    def get_analog_combiner(self):
        return torch.exp(1j * self.analog_phases)  # [Nr, NRF]
    
    def get_Q(self):
        QA = self.get_analog_combiner().to(torch.complex64)         # [Nr, NRF]
        QD = self.digital_combiner.to(torch.complex64)              # [K, NRF, Ns]
        Q = torch.matmul(QA, QD)                                    # Overall combiner: [Nr, Ns]
        return Q
    
    def forward(self, y_t):

        # Apply hybrid combiner: y[k] = Q[k]^H @ r[k]
        QH = self.get_Q().conj().transpose(1, 2)              # [K, Ns, Nr]
        s_hat_t = torch.matmul(QH, y_t.unsqueeze(-1)).squeeze(-1)  # [K, Ns]
        return s_hat_t

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
        self.M = config.M # Modulation order
        self.Nt = config.Nt
        self.Nr = config.Nr
        self.NRF = config.NRF
        self.Ns = config.Ns
        self.num_subcarriers = config.num_subcarriers
        self.noise_power = config.noise_power  # scalar noise power
        
        # Channel parameters (for a geometry-based channel simulation)
        self.num_clusters = config.num_clusters
        self.num_rays = config.num_rays

        # Initialize modulation model
        self.mary_modulation = MAryModulation(M=self.M)
        
        # Initialize hybrid precoder and combiner modules
        self.precoder = HybridPrecoderModule(self.Nt, self.NRF, self.Ns, self.num_subcarriers)
        self.combiner = HybridCombinerModule(self.Nr, self.NRF, self.Ns, self.num_subcarriers)

        # Initialize geometry-based channel model
        self.channel = GeometryBasedChannel(self.Nt, self.Nr, self.num_clusters, self.num_rays, device=device, subcarriers=self.num_subcarriers)

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

        # Permute to [B, length, L, group, 2]
        x = modulated_symbols.permute(1, 2, 0, 3, 4)  # [B, len, L, group, 2]

        # Reshape: [B * len, L * group, 2] → [T, K, Ns]
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

        Args:
            modulated_grid: [T, K, Ns] - sequence of modulated features

        Returns:
            recovered_signal: [T, K, Ns] - after channel and beamforming
        """
        modulated_symbols = self.mary_modulation.modulate(symbols) #  [L, B, len, group, 2] complex
        
        # modulated_grid: [T, K, Ns] where T = B * len, K = L * group, Ns = 2
        modulated_grid = self.reshape_modulated_symbols_to_grid(modulated_symbols)  # [T, K, Ns]

        T, K, Ns = modulated_grid.shape
        assert K == self.num_subcarriers


        output_sequence = []

        for t in range(T):
            s_t = modulated_grid[t]                        # [K, Ns]
            s_t = s_t.unsqueeze(-1)                        # [K, Ns, 1]

            # === Per-time step: update channel ===
            self.channel.update_channel()                  # [K, Nr, Nt]  
            H = self.channel.get_channel()                 # [K, Nr, Nt]
            
            # === Apply hybrid precoder ===
            x_t = self.precoder(s_t)                       # [K, Nt]

            # Apply channel: y[k] = H[k] @ x[k]
            y_t = self.channel(x_t.unsqueeze(-1))     # [K, Nr]   

            # === Apply hybrid combiner ===
            s_hat_t = self.combiner(y_t).squeeze(0)       # [K, Ns]
            output_sequence.append(s_hat_t)

        output_grid = torch.stack(output_sequence, dim=0)  # [T, K, Ns]
        
        # Reshape output_grid: [T, K, Ns] → [L, B, len, group, 2]
        output_sequence = self.reshape_grid_to_modulated_sequence(output_grid, modulated_symbols.shape)  # [L, B, len, group, 2]

        # Apply modulation demodulation
        demodulate_symbol = self.mary_modulation.demodulate(output_sequence)
        return demodulate_symbol                               # [T, K, Ns]
        
    def WMMSE(self, s_t, Weight=None, num_iter=10):
        
        Weight = Weight if Weight is not None else torch.full(
            (self.K,), 1.0 / self.K, dtype=torch.float32, device=s_t.device
        )
        # We now perform per-subcarrier beamforming.
        V_list = []  # List to hold overall precoders for each subcarrier, shape [B, Nt, Ns]
        Q_list = []  # List to hold overall combiners for each subcarrier, shape [B, Nr, Ns]
        
        # For each subcarrier k, we optimize beamforming separately (here we do a few iterations)
        for k in range(K):
            V_sub = []
            Q_sub = []
            # Get channel for batch element b and subcarrier k: [Nr, Nt]
            H_bk = H[b, k, :, :]
            # --- Digital update (closed-form) ---
            VA = self.precoder.get_analog_precoder()  # [Nt, NRF]
            VD = update_digital_precoder(H_bk, VA, self.noise_power)  # [NRF, Ns]
            # Update digital precoder (detached from gradient; this optimization is separate)
            self.precoder.digital_precoder.data.copy_(VD)
            V_opt = torch.matmul(VA, VD)  # [Nt, Ns]
            V_sub.append(V_opt)
            
            QA = self.combiner.get_analog_combiner()  # [Nr, NRF]
            QD = update_digital_combiner(H_bk, QA, self.noise_power)  # [NRF, Ns]
            self.combiner.digital_combiner.data.copy_(QD)
            Q_opt = torch.matmul(QA, QD)  # [Nr, Ns]
            Q_sub.append(Q_opt)
            V_sub = torch.stack(V_sub, dim=0)  # [B, Nt, Ns]
            Q_sub = torch.stack(Q_sub, dim=0)  # [B, Nr, Ns]
            V_list.append(V_sub)
            Q_list.append(Q_sub)
        
        # Stack the precoders and combiners along subcarrier dimension:
        # V_all: [B, K, Nt, Ns], Q_all: [B, K, Nr, Ns]
        V_all = torch.stack(V_list, dim=1)
        Q_all = torch.stack(Q_list, dim=1)
        
        # Now, for each subcarrier, transmit the symbols:
        # modulated_grid: [B, K, NS] -> unsqueeze last dim to [B, K, NS, 1]
        S = modulated_grid.unsqueeze(-1)
        # For each subcarrier k, the transmitted signal is: X = V[k] * S
        # (Perform batch-matrix multiplication for each subcarrier)
        X = torch.matmul(V_all, S)  # [B, K, Nt, 1]
        X = X.squeeze(-1)  # [B, K, Nt]
        
        # Apply the channel: Y = H * X + noise. For each subcarrier, we treat X as [B, K, Nt, 1]
        X_exp = X.unsqueeze(-1)  # [B, K, Nt, 1]
        Y = torch.matmul(H, X_exp)  # [B, K, Nr, 1]
        Y = Y.squeeze(-1)  # [B, K, Nr]
        
        # Add AWGN noise (complex Gaussian)
        noise = (torch.sqrt(self.noise_power / 2) * 
                 (torch.randn_like(Y) + 1j * torch.randn_like(Y)))
        Y = Y + noise  # [B, K, Nr]
        
        # Receiver combining: for each subcarrier, s_hat = Q^H * Y.
        # First, unsqueeze Y: [B, K, Nr, 1]
        Y_exp = Y.unsqueeze(-1)
        # Compute Q^H: for each subcarrier, Q_all: [B, K, Nr, Ns] -> QH: [B, K, Ns, Nr]
        QH = Q_all.conj().permute(0, 1, 3, 2)
        s_hat = torch.matmul(QH, Y_exp)  # [B, K, Ns, 1]
        s_hat = s_hat.squeeze(-1)  # [B, K, Ns]
        
        return s_hat

#########################################
# Helper functions for digital updates
#########################################
def update_digital_precoder(H, VA, noise_power):
    """
    Given channel H [Nr, Nt] and analog precoder VA [Nt, NRF],
    update digital precoder VD using an MMSE–like update:
    
    VD = (VA^H H^H H VA + σ^2 I)^{-1} (VA^H H^H)
    """
    NRF = VA.shape[1]
    I = torch.eye(NRF, dtype=VA.dtype, device=VA.device)
    A = torch.matmul(VA.conj().t(), torch.matmul(H.conj().t(), H) @ VA)
    VD = torch.linalg.solve(A + noise_power * I, torch.matmul(VA.conj().t(), H.conj().t()))
    return VD

def update_digital_combiner(H, QA, noise_power):
    """
    Given channel H [Nr, Nt] and analog combiner QA [Nr, NRF],
    update digital combiner QD:
    
    QD = (QA^H H H^H QA + σ^2 I)^{-1} (QA^H H)
    """
    NRF = QA.shape[1]
    I = torch.eye(NRF, dtype=QA.dtype, device=QA.device)
    A = torch.matmul(QA.conj().t(), H @ H.conj().t() @ QA)
    QD = torch.linalg.solve(A + noise_power * I, torch.matmul(QA.conj().t(), H))
    return QD

