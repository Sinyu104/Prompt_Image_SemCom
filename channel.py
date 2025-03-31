import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryBasedChannel(nn.Module):
    def __init__(self, Nt, Nr, NC, NR):
        """
        Nt: number of transmit antennas
        Nr: number of receive antennas
        NC: number of clusters
        NR: number of rays per cluster
        """
        super(GeometryBasedChannel, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.NC = NC
        self.NR = NR

    def array_response_tx(self, theta):
        # Transmit array steering vector (uniform linear array)
        n = torch.arange(self.Nt, dtype=torch.float32, device=theta.device)
        return (1/math.sqrt(self.Nt)) * torch.exp(1j * math.pi * n * torch.sin(theta))

    def array_response_rx(self, theta):
        # Receive array steering vector
        n = torch.arange(self.Nr, dtype=torch.float32, device=theta.device)
        return (1/math.sqrt(self.Nr)) * torch.exp(1j * math.pi * n * torch.sin(theta))

    def forward(self, k=0):
        """
        Generate channel matrix H for one subcarrier.
        According to the paper:
        H = sqrt(Nt*Nr/(NC*NR)) * sum_{c=1}^{NC} sum_{l=1}^{NR} α_{cl} a_r(θ_{r_{cl}}) a_t(θ_{t_{cl}})^H
        """
        H = 0.0
        scaling = math.sqrt(self.Nt * self.Nr / (self.NC * self.NR))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for c in range(self.NC):
            for l in range(self.NR):
                # Complex gain: Rayleigh fading (zero mean, unit variance)
                alpha = (torch.randn(1, device=device) + 1j * torch.randn(1, device=device)) / math.sqrt(2)
                # Random departure/arrival angles (e.g., uniformly drawn in a certain range)
                theta_t = (torch.rand(1, device=device) - 0.5) * math.pi/2  # from -pi/4 to pi/4
                theta_r = (torch.rand(1, device=device) - 0.5) * math.pi/2
                a_t = self.array_response_tx(theta_t)  # shape [Nt]
                a_r = self.array_response_rx(theta_r)  # shape [Nr]
                # Outer product gives a matrix of shape [Nr, Nt]
                H += alpha * torch.ger(a_r, a_t.conj())
        H = scaling * H
        return H  # shape [Nr, Nt], complex

#####################################
# 2. Hybrid Precoder Module         #
#####################################
class HybridPrecoderModule(nn.Module):
    def __init__(self, Nt, NRF, Ns):
        """
        Nt: number of transmit antennas
        NRF: number of RF chains (analog precoder dimension)
        Ns: number of data streams (digital precoder output dimension)
        """
        super(HybridPrecoderModule, self).__init__()
        self.Nt = Nt
        self.NRF = NRF
        self.Ns = Ns

        # Analog precoder: parameterize by phases to satisfy |[VA]ij|=1.
        # We will store learnable phases of shape [Nt, NRF].
        self.analog_phases = nn.Parameter(torch.rand(Nt, NRF) * 2 * math.pi)
        # Digital precoder: a learnable linear mapping from NRF to Ns.
        self.digital_precoder = nn.Linear(NRF, Ns, bias=False)

    def forward(self, S):
        """
        S: input symbol vector, shape [B, Ns] (assume complex tensor)
        Returns: transmitted signal X [B, Nt] and combined precoder V [Nt, Ns]
        """
        # Construct analog precoder VA: exp(j*phase)
        VA = torch.exp(1j * self.analog_phases)  # shape [Nt, NRF]
        # Get digital precoder weight (note: weight shape is [Ns, NRF])
        # We want a digital matrix VD of shape [NRF, Ns].
        VD = self.digital_precoder.weight.transpose(0, 1)  # shape [NRF, Ns]
        # Combined precoder: V = VA * VD, shape [Nt, Ns]
        V = torch.matmul(VA, VD)
        # Transmit: X = V * s (for each sample, s has shape [Ns])
        X = torch.matmul(S, V.transpose(0, 1))  # [B, Nt]
        return X, V

#####################################
# 3. Hybrid Combiner Module         #
#####################################
class HybridCombinerModule(nn.Module):
    def __init__(self, Nr, NRF, Ns):
        """
        Nr: number of receive antennas
        NRF: number of RF chains at receiver
        Ns: number of data streams (digital combiner output dimension)
        """
        super(HybridCombinerModule, self).__init__()
        self.Nr = Nr
        self.NRF = NRF
        self.Ns = Ns

        # Analog combiner: learnable phases
        self.analog_phases = nn.Parameter(torch.rand(Nr, NRF) * 2 * math.pi)
        # Digital combiner: learnable linear mapping from NRF to Ns
        self.digital_combiner = nn.Linear(NRF, Ns, bias=False)

    def forward(self, Y):
        """
        Y: received signal, shape [B, Nr]
        Returns: combined output s_hat, and overall combiner Q
        """
        QA = torch.exp(1j * self.analog_phases)  # [Nr, NRF]
        QD = self.digital_combiner.weight.transpose(0, 1)  # [NRF, Ns]
        Q = torch.matmul(QA, QD)  # overall combiner, shape [Nr, Ns]
        # Apply combiner: s_hat = Q^H Y^T, then transpose back
        s_hat = torch.matmul(Y, Q)  # [B, Ns]
        return s_hat, Q

#####################################
# 4. Hybrid Beamforming Module      #
#####################################
class HybridBeamformingModule(nn.Module):
    def __init__(self, Nt, Nr, NRF, Ns):
        super(HybridBeamformingModule, self).__init__()
        self.precoder = HybridPrecoderModule(Nt, NRF, Ns)
        self.combiner = HybridCombinerModule(Nr, NRF, Ns)

    def forward(self, S, H):
        """
        S: transmitted symbols, shape [B, Ns] (complex tensor)
        H: channel matrix, shape [Nr, Nt] (complex tensor)
        """
        # Transmitter: apply precoder
        X, V = self.precoder(S)  # X: [B, Nt]
        # Channel: simulate transmission: Y = H * X for each sample.
        # Assume H is same for the batch (or use batch channel if available)
        Y = torch.matmul(H, X.transpose(0,1)).transpose(0,1)  # [B, Nr]
        # Receiver: apply combiner
        s_hat, Q = self.combiner(Y)  # [B, Ns]
        return s_hat, V, Q

    
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
        print("constellation: ", self.constellation)

    def _generate_constellation(self, M):
        """
        Generate M points evenly spaced on the unit circle in the complex plane.
        Returns complex-valued tensor of shape [M].
        """
        angles = 2 * torch.pi * torch.arange(M) / M
        constellation = torch.exp(1j * angles)
        return constellation  # [M]

    def forward(self, x):
        """
        Args:
            x: Long tensor of shape [len], values in [0, 63]
        Returns:
            modulated: [len, 2], each entry is complex number (2 symbols per input)
        """
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        assert x.dtype == torch.long and x.max() < 64
        shape = x.shape  # [B, len, group]
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
        return modulated.view(*shape, 2)  # complex tensor

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
# 1. Hybrid Precoder/Combiner Modules
##########################################
class HybridPrecoderModule(nn.Module):
    def __init__(self, Nt, NRF, Ns):
        """
        Nt: number of transmit antennas
        NRF: number of RF chains (analog precoder dimension)
        Ns: number of data streams (digital precoder output dimension)
        """
        super(HybridPrecoderModule, self).__init__()
        self.Nt = Nt
        self.NRF = NRF
        self.Ns = Ns

        # Instead of storing the full complex matrix, we parameterize
        # the analog precoder via real-valued phases.
        self.analog_phases = nn.Parameter(torch.rand(Nt, NRF) * 2 * math.pi)
        # Digital precoder: unconstrained complex matrix (learned via closed‐form update)
        # We initialize it randomly.
        self.digital_precoder = nn.Parameter(torch.randn(NRF, Ns, dtype=torch.cfloat))

    def get_analog_precoder(self):
        # Construct the analog precoder from phases:
        return torch.exp(1j * self.analog_phases)  # shape: [Nt, NRF]

    def forward(self, digital_input=None):
        # For our optimization routine, the module output is just the overall precoder:
        VA = self.get_analog_precoder()  # [Nt, NRF]
        VD = self.digital_precoder         # [NRF, Ns]
        V = torch.matmul(VA, VD)            # Overall precoder: [Nt, Ns]
        return V, VA, VD

class HybridCombinerModule(nn.Module):
    def __init__(self, Nr, NRF, Ns):
        """
        Nr: number of receive antennas
        NRF: number of RF chains at receiver
        Ns: number of data streams (digital combiner output dimension)
        """
        super(HybridCombinerModule, self).__init__()
        self.Nr = Nr
        self.NRF = NRF
        self.Ns = Ns

        self.analog_phases = nn.Parameter(torch.rand(Nr, NRF) * 2 * math.pi)
        self.digital_combiner = nn.Parameter(torch.randn(NRF, Ns, dtype=torch.cfloat))

    def get_analog_combiner(self):
        return torch.exp(1j * self.analog_phases)  # [Nr, NRF]

    def forward(self):
        QA = self.get_analog_combiner()  # [Nr, NRF]
        QD = self.digital_combiner       # [NRF, Ns]
        Q = torch.matmul(QA, QD)          # [Nr, Ns]
        return Q, QA, QD

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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#############################
# Hybrid Precoders & Combiners
#############################

class HybridPrecoderModule(nn.Module):
    def __init__(self, Nt, NRF, Ns):
        """
        Nt: number of transmit antennas
        NRF: number of RF chains
        Ns: number of data streams per subcarrier
        """
        super(HybridPrecoderModule, self).__init__()
        self.Nt = Nt
        self.NRF = NRF
        self.Ns = Ns
        # Parameterize analog precoder by phases (real-valued)
        self.analog_phases = nn.Parameter(torch.rand(Nt, NRF) * 2 * math.pi)
        # Digital precoder (unconstrained complex)
        self.digital_precoder = nn.Parameter(torch.randn(NRF, Ns, dtype=torch.cfloat))
    
    def get_analog_precoder(self):
        # Constant-modulus analog precoder: exp(j*phase)
        return torch.exp(1j * self.analog_phases)  # shape: [Nt, NRF]
    
    def forward(self):
        VA = self.get_analog_precoder()         # [Nt, NRF]
        VD = self.digital_precoder              # [NRF, Ns]
        V = torch.matmul(VA, VD)                # Overall precoder: [Nt, Ns]
        return V, VA, VD

class HybridCombinerModule(nn.Module):
    def __init__(self, Nr, NRF, Ns):
        """
        Nr: number of receive antennas
        NRF: number of RF chains
        Ns: number of data streams per subcarrier
        """
        super(HybridCombinerModule, self).__init__()
        self.Nr = Nr
        self.NRF = NRF
        self.Ns = Ns
        self.analog_phases = nn.Parameter(torch.rand(Nr, NRF) * 2 * math.pi)
        self.digital_combiner = nn.Parameter(torch.randn(NRF, Ns, dtype=torch.cfloat))
    
    def get_analog_combiner(self):
        return torch.exp(1j * self.analog_phases)  # [Nr, NRF]
    
    def forward(self):
        QA = self.get_analog_combiner()         # [Nr, NRF]
        QD = self.digital_combiner              # [NRF, Ns]
        Q = torch.matmul(QA, QD)                # Overall combiner: [Nr, Ns]
        return Q, QA, QD

#########################################
# Physical Layer Module
#########################################
class PhysicalLayerModule(nn.Module):
    def __init__(self, config):
        """
        config should contain:
          Nt, Nr, NRF, Ns, num_subcarriers, noise_power, and channel parameters.
        """
        super(PhysicalLayerModule, self).__init__()
        self.Nt = config.Nt
        self.Nr = config.Nr
        self.NRF = config.NRF
        self.Ns = config.Ns
        self.num_subcarriers = config.num_subcarriers
        self.noise_power = config.noise_power  # scalar noise power
        
        # Channel parameters (for a geometry-based channel simulation)
        self.num_clusters = config.get("num_clusters", 3)
        self.num_rays = config.get("num_rays", 5)
        
        # Initialize hybrid precoder and combiner modules
        self.precoder = HybridPrecoderModule(self.Nt, self.NRF, self.Ns)
        self.combiner = HybridCombinerModule(self.Nr, self.NRF, self.Ns)
    
    def simulate_channel(self, B):
        """
        Simulate a geometry-based multi-path fading channel.
        For each subcarrier and for each sample in the batch, generate a channel H.
        Output shape: [B, num_subcarriers, Nr, Nt]
        
        Here we use a simple random complex Gaussian channel as a placeholder.
        """
        H_real = torch.randn(B, self.num_subcarriers, self.Nr, self.Nt, device=self.precoder.analog_phases.device)
        H_imag = torch.randn(B, self.num_subcarriers, self.Nr, self.Nt, device=self.precoder.analog_phases.device)
        H = H_real + 1j * H_imag
        # (A more detailed geometry-based channel model can be inserted here.)
        return H
    
    def forward(self, modulated_grid):
        """
        modulated_grid: [B, K, NS] where K = num_subcarriers, NS = symbols per subcarrier.
        Returns: recovered signal after beamforming and channel transmission, shape [B, K, NS]
        """
        B, K, NS = modulated_grid.shape
        assert K == self.num_subcarriers, "Mismatch between grid subcarriers and config."
        
        # Simulate channel for each sample and subcarrier
        H = self.simulate_channel(B)  # [B, K, Nr, Nt]
        
        # We now perform per-subcarrier beamforming.
        V_list = []  # List to hold overall precoders for each subcarrier, shape [B, Nt, Ns]
        Q_list = []  # List to hold overall combiners for each subcarrier, shape [B, Nr, Ns]
        
        # For each subcarrier k, we optimize beamforming separately (here we do a few iterations)
        for k in range(K):
            V_sub = []
            Q_sub = []
            for b in range(B):
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

