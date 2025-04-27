import numpy as np
from PIL import Image
import io
import os
import torch
from torch import nn
from model import LLavaAnswerGenerationModel
from utils import store_generated_outputs
from torchvision.transforms import ToPILImage, ToTensor
from types import MethodType

import matplotlib.pyplot as plt

def save_image_comparison(original, reconstructed, save_path, image_id, title=None):

    """
    Display original and reconstructed images side-by-side.
    
    Args:
        original (torch.Tensor): Original image (C, H, W) in [0, 1].
        reconstructed (torch.Tensor): Reconstructed image (C, H, W) in [0, 1].
    """
    original_np = original.permute(1, 2, 0).cpu().numpy()
    reconstructed_np = reconstructed.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 5))
    if title:
        plt.suptitle(title, fontsize=14)

    plt.subplot(1, 2, 1)
    plt.imshow(original_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_np)
    plt.title("Received Image")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"compare_{image_id}.png"))

def save_reconstructed_image(reconstructed, save_path, image_id):
    """
    Save the reconstructed image as a standalone file.

    Args:
        reconstructed (torch.Tensor): Reconstructed image tensor (C, H, W) in [0, 1].
        save_path (str): Directory to save the image.
        image_id (int or str): Identifier for the image.
    """
    reconstructed_np = reconstructed.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(5, 5))
    plt.imshow(reconstructed_np)
    plt.axis("off")
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f"reconstructed_{image_id}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()


def encode_tensor_to_jpeg2000(image_tensor, quality_mode='rates', quality_val=1.0):
    """
    Encode an image tensor to JPEG 2000 bytes using PIL (OpenJPEG backend).
    
    Args:
        image_tensor (torch.Tensor): Image tensor (C, H, W) in [0, 1]
        quality_mode (str): 'rates' for bitrate control or 'dB' for PSNR control
        quality_val (float): Value depending on mode. e.g., rate=1.0 or psnr=40.0
        
    Returns:
        bytes: JPEG 2000 encoded image
    """
    pil_image = ToPILImage()(image_tensor.detach().cpu())
    buf = io.BytesIO()
    
    pil_image.save(buf, format="JPEG2000", quality_mode=quality_mode, quality_layers=[quality_val])
    return buf.getvalue()


def jpeg_bytes_to_bits(jpeg_bytes):
    """
    Convert JPEG bytes into a numpy array of bits.
    
    Args:
        jpeg_bytes (bytes): The JPEG file bytes.
        
    Returns:
        numpy.ndarray: A 1D array of bits (0s and 1s).
    """
    data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bits = np.unpackbits(data)
    return bits

def bits_to_jpeg_bytes(bits):
    """
    Pack a numpy array of bits back into JPEG bytes.
    
    Args:
        bits (numpy.ndarray): A 1D array of bits.
        
    Returns:
        bytes: Reconstructed JPEG bytes.
    """
    data = np.packbits(bits)
    jpeg_bytes = data.tobytes()
    return jpeg_bytes

def qam8_modulation(bits):
    """
    Modulate bits using 8-QAM. Pads bits if needed to make length divisible by 3.

    Returns:
        symbols: complex-valued modulated symbols
        pad_len: number of bits added as padding (to remove later)
    """
    bits = bits.astype(np.uint8)
    bits = np.clip(bits, 0, 1)

    pad_len = (3 - (len(bits) % 3)) % 3
    if pad_len > 0:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])

    bit_triplets = bits.reshape(-1, 3)
    decimal = bit_triplets[:, 0] * 4 + bit_triplets[:, 1] * 2 + bit_triplets[:, 2]

    const_map = {
        0: -1 - 1j,
        1: -1 + 0j,
        2: -1 + 1j,
        3:  0 - 1j,
        4:  0 + 1j,
        5:  1 - 1j,
        6:  1 + 0j,
        7:  1 + 1j,
    }

    symbols = np.array([const_map[d] for d in decimal], dtype=np.complex128)
    return symbols, pad_len



def qam8_demodulation(symbols, pad_len=0):
    """
    Demodulate 8-QAM symbols back to bits, removing `pad_len` bits from the end.

    Returns:
        bits: recovered bit array with padding removed
    """
    const_map = {
        0: -1 - 1j,
        1: -1 + 0j,
        2: -1 + 1j,
        3:  0 - 1j,
        4:  0 + 1j,
        5:  1 - 1j,
        6:  1 + 0j,
        7:  1 + 1j,
    }
    symbol_list = np.array(list(const_map.values()))
    index_list = np.array(list(const_map.keys()))

    recovered_bits = []
    for s in symbols:
        dists = np.abs(s - symbol_list)
        nearest_idx = index_list[np.argmin(dists)]
        bits = [int(b) for b in format(nearest_idx, '03b')]
        recovered_bits.extend(bits)

    if pad_len > 0:
        recovered_bits = recovered_bits[:-pad_len]

    return np.array(recovered_bits, dtype=np.uint8)



def simulate_flat_fading_channel(symbols, snr_db):
    """
    Simulate a flat fading channel with complex AWGN on complex symbols.
    
    A complex Rayleigh fading coefficient (h) is used, and complex AWGN noise
    is added based on the given SNR (in dB). The receiver is assumed to have
    perfect channel knowledge and equalizes by dividing out h.
    
    Args:
        symbols (np.ndarray): Transmitted complex symbols (1D array, dtype=complex).
        snr_db (float): Signal-to-noise ratio in dB.
    
    Returns:
        tuple: (equalized received symbols, complex channel coefficient h)
    """
    # Complex Rayleigh fading: h = h_real + j*h_imag, where h_real, h_imag ~ N(0, 0.5)
    h_real = np.random.normal(0, np.sqrt(0.5))
    h_imag = np.random.normal(0, np.sqrt(0.5))
    h = h_real + 1j * h_imag  # Complex fading coefficient

    snr_linear = 10 ** (snr_db / 10)

    # Noise variance per dimension = 1 / (2 * SNR)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    noise = np.random.normal(0, noise_std, symbols.shape) + 1j * np.random.normal(0, noise_std, symbols.shape)

    # Transmit over channel
    y = h * symbols + noise

    # Equalization (perfect CSI)
    equalized = y / h

    return equalized, h



def decode_jpeg2000_from_bytes(jp2_bytes):
    """
    Decode JPEG 2000 bytes into a PIL Image.
    
    Args:
        jp2_bytes (bytes): JPEG 2000 byte stream.
    
    Returns:
        PIL.Image or None
    """
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    try:
        buf = io.BytesIO(jp2_bytes)
        img = Image.open(buf)
        img.load()
        return img
    except Exception as e:
        print("Failed to decode JPEG2000 image:", e)
        return None

    

class JPGTransmission(nn.Module):
    def __init__(self, args, device=None):
        super(JPGTransmission, self).__init__()
        self.device = device
        self.snr = args.SNR  # Use snr_db from args
        self.quality = 95
        self.store_gen_data = args.store_gen_data
    
        if self.store_gen_data:
            os.makedirs(args.generated_data_dir, exist_ok=True)
            self.generated_data_dir = args.generated_data_dir

        self.perceptual_model = LLavaAnswerGenerationModel(
            model_path="qnguyen3/nanoLLaVA-1.5",
            device=self.device
        ).to(self.device)

        self.perceptual_model.eval()

        # Reconstruction Loss (e.g., L1 Loss)
        self.recon_loss = nn.L1Loss()

    def forward(self, input_image, question, answer, image_id):
         # ---- JPEG2000 Encoding Stage ----
        jpeg_bytes = encode_tensor_to_jpeg2000(input_image)

        # ---- Split header and body ----
        header_size = 200  # You can tune this (JP2 header + codestream marker segments)
        header_bytes = jpeg_bytes[:header_size]
        body_bytes = jpeg_bytes[header_size:]

        # ---- Convert only body to bits ----
        body_bits = jpeg_bytes_to_bits(body_bytes)

        # ---- QAM Modulation Stage ----
        symbols, pad_len = qam8_modulation(body_bits)

        # ---- Transmission (Flat Fading Channel) ----
        received_symbols, h = simulate_flat_fading_channel(symbols, self.snr)

        # ---- Demodulation ----
        recovered_bits = qam8_demodulation(received_symbols, pad_len=pad_len)

        # ---- BER Calculation ----
        bit_errors = np.sum(body_bits != recovered_bits)
        ber = bit_errors / len(body_bits)


        # ---- Reconstruct JPEG Bytes ----
        recovered_body_bytes = bits_to_jpeg_bytes(recovered_bits)
        recovered_jpeg_bytes = header_bytes + recovered_body_bytes

        # ---- JPEG2000 Decoding ----
        received_image = decode_jpeg2000_from_bytes(recovered_jpeg_bytes)
        if received_image is not None:
            received_image = ToTensor()(received_image)
            if torch.mean(received_image) < 1e-3:
                print(f"Image {image_id} decoded as black. Likely due to data corruption.")

        else:
            received_image = torch.zeros_like(input_image)
            print("Failed to decode received image, reverting to black image.")

        # Optionally, store generated outputs if desired.
        if self.store_gen_data:
            save_reconstructed_image(received_image, self.generated_data_dir, image_id)
            # save_image_comparison(input_image, received_image, self.generated_data_dir, image_id)
        
        # Compute perceptual loss between the original and received images.
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        if received_image.dim() == 3:
            received_image = received_image.unsqueeze(0)
        received_image = received_image.to(input_image.device)
        with torch.no_grad():
            perc_loss  = self.perceptual_model(input_image, received_image, [question])["loss_perc"]

        return perc_loss
