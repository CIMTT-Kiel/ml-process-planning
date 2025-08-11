import torch
import torch.nn as nn
import torch.nn.functional as F

#custom imports
from cadtoseq.constants import VOCAB
from cadtoseq.ml.datasets.fabricad import Fabricad

# AutoRegressiveManufacturingStepTransformerDecoder(ARMSTD) 
class ARMSTD(nn.Module):
    def __init__(self, input_dim=32, set_size=1024, embed_dim=512, num_steps=VOCAB.__len__(), max_seq_len=6, num_layers=6, nhead=8, dropout=0.1):
        super().__init__()

        self.input_linear = nn.Linear(input_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512, batch_first=True, dropout=dropout, 
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.step_embeddings = nn.Embedding(num_steps, embed_dim)
        self.output_linear = nn.Linear(embed_dim, num_steps)

        self.max_seq_len = max_seq_len
        self.num_steps = num_steps
        self.stop_token_id = VOCAB["STOP"]  

        self.input_dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, vector_set, tgt_seq):
        batch_size = vector_set.size(0)
        memory = self.input_dropout(self.input_linear(vector_set)) 

        tgt_embedded = self.embedding_dropout(self.step_embeddings(tgt_seq))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embedded.size(1)).to(vector_set.device)
        tgt_key_padding_mask = tgt_seq == VOCAB["PAD"]

        output = self.decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        logits = self.output_linear(output)
        return logits

    def generate(self, vector_set, return_probs=False, device="cpu"):
        batch_size = vector_set.size(0)
        vector_set = vector_set.to(device)
        memory = self.input_linear(vector_set) 

        generated = torch.full((batch_size, 1), VOCAB["START"], dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        all_probs = []

        for _ in range(self.max_seq_len):
            tgt_embedded = self.step_embeddings(generated)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embedded.size(1)).to(device)

            output = self.decoder(
                tgt=tgt_embedded,
                memory=memory,
                tgt_mask=tgt_mask

            )

            logits = self.output_linear(output[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.unsqueeze(1))

            next_token = torch.argmax(probs, dim=-1).to(device)
            
            next_token[finished] = VOCAB["PAD"]

            finished |= (next_token == self.stop_token_id)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # add padding if for one or more sequences the stop token was reached or 
        if generated.size(1) < self.max_seq_len + 1:  # the plus one is because of the manual added START token
            pad_len = self.max_seq_len + 1 - generated.size(1)
            pad = torch.full((batch_size, pad_len), VOCAB["PAD"], dtype=torch.long, device=device)
            generated = torch.cat([generated, pad], dim=1)

        all_probs = torch.cat(all_probs, dim=1) if return_probs else None

        # Remove the START token from the generated sequence
        return (generated[:, 1:], all_probs) if return_probs else generated[:, 1:]
    
    def train_model():
        pass  

if __name__ == "__main__":
    batch_size = 1
    vector_set = torch.randn(batch_size, 1024, 32)

    model = ARMSTD()
    generated_seq = model.generate(vector_set, device=vector_set.device)

    print(Fabricad.decode_sequence(generated_seq[0].tolist()))

    print("Generated sequence:", [len(seq) for seq in generated_seq])
