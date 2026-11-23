import torch
import torch.nn as nn

class FeatureModuleEncoderTest(nn.Module):
    """
    Az EINN FeatureModuleEncoder leegyszerűsített másolata, amely 
    képes ki- és bekapcsolni az Attention réteg maszkolását.
    """
    def __init__(self, dim_seq_in=5, rnn_out=40):
        super().__init__()
        self.enc_rnn = nn.GRU(dim_seq_in, rnn_out, batch_first=True)
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=rnn_out, 
            num_heads=1, 
            batch_first=True
        )
        
        self.enc_out_layer = nn.Sequential(
            nn.Linear(rnn_out, rnn_out),
            nn.Tanh()
        )

    def forward(self, x, mask, apply_attn_mask=False):
        latent_seqs, _ = self.enc_rnn(x)

        # 1. ATTENTION MASZKOLÁS BEÁLLÍTÁSA
        # A PyTorch key_padding_mask paramétere True-t vár a paddingre, False-ot a valósra.
        # Mivel a te maszkod: 1.0 = padding, 0.0 = valós, a .bool() konverzió tökéletes.
        key_padding_mask = mask.bool() if apply_attn_mask else None

        # Figyelem réteg futtatása (opcionális maszkkal)
        latent_seqs, _ = self.attn_layer(
            query=latent_seqs,
            key=latent_seqs,
            value=latent_seqs,
            key_padding_mask=key_padding_mask
        )

        # 2. MANUÁLIS MASZKOLÁS (Ez a te meglévő kódod, ami a kimeneti átlagolást védi)
        valid_mask = (1 - mask).unsqueeze(-1)
        latent_seqs *= valid_mask
        h = latent_seqs.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        h = self.enc_out_layer(h)
        return h


def run_attention_mask_test():
    torch.manual_seed(42)
    
    d_x = 5
    rnn_out = 40
    real_days = 40

    # 1. Generálunk 40 napnyi "tiszta", valódi adatot
    x_real = torch.randn(real_days, d_x)

    # 2. Létrehozzuk a három különböző ablakméretet
    lengths = [200, 120, 80]
    x_padded_list = []
    mask_list = []

    for total_len in lengths:
        # Bemenet paddingelése 0.0-val a végéig
        x_pad = torch.zeros(1, total_len, d_x)
        x_pad[0, :real_days, :] = x_real

        # Maszk elkészítése (0.0 = valós, 1.0 = padding)
        m = torch.ones(1, total_len)
        m[0, :real_days] = 0.0

        x_padded_list.append(x_pad)
        mask_list.append(m)

    # Modell inicializálása és Eval módba tétele (hogy a Dropout ne okozzon zajt a tesztben)
    encoder = FeatureModuleEncoderTest(dim_seq_in=d_x, rnn_out=rnn_out)
    encoder.eval() 

    # --- TESZT 1: MEGLÉVŐ ÁLLAPOT (NINCS ATTENTION MASZK) ---
    print("--- TESZT 1: ATTENTION MASZK NÉLKÜL (Jelenlegi állapot) ---")
    h_no_mask = []
    with torch.no_grad():
        for x, m in zip(x_padded_list, mask_list):
            h = encoder(x, m, apply_attn_mask=False)
            h_no_mask.append(h)

    print(f"Euklideszi távolság (200 hosszú vs 120 hosszú): {torch.dist(h_no_mask[0], h_no_mask[1]).item():.6f}")
    print(f"Euklideszi távolság (200 hosszú vs 80 hosszú) : {torch.dist(h_no_mask[0], h_no_mask[2]).item():.6f}")
    print("Következtetés: A reprezentáció drasztikusan elcsúszik a padding hossza miatt!\n")

    # --- TESZT 2: JAVÍTOTT ÁLLAPOT (ATTENTION MASZKKAL) ---
    print("--- TESZT 2: ATTENTION MASZKKAL (Javasolt állapot) ---")
    h_with_mask = []
    with torch.no_grad():
        for x, m in zip(x_padded_list, mask_list):
            h = encoder(x, m, apply_attn_mask=True)
            h_with_mask.append(h)

    print(f"Euklideszi távolság (200 hosszú vs 120 hosszú): {torch.dist(h_with_mask[0], h_with_mask[1]).item():.6f}")
    print(f"Euklideszi távolság (200 hosszú vs 80 hosszú) : {torch.dist(h_with_mask[0], h_with_mask[2]).item():.6f}")
    print("Következtetés: A reprezentáció betonstabil, a távolság 0.000000. A hiba megszűnt!")

if __name__ == "__main__":
    run_attention_mask_test()
