import torch 

import torch.nn as nn   


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: None):
        """
        Args:
            theta: The theta parameter for the RoPE
            d_k: The dimension of the key and query
            max_seq_len: The maximum sequence length
            device: The device to use
            dtype: The dtype to use
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
     
    def rotation_block(self, theta: float, block_index: int, seq_pos: int, d_k: int) -> torch.Tensor:

        angle = torch.tensor(seq_pos/ (theta **(2*block_index/d_k)))
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        r_matrix = torch.tensor([[cos, -sin], [sin, cos]])
        return r_matrix


    def get_rotation_matrix(self, theta: float, d_k: int, max_seq_len: int) -> torch.Tensor:

        rotation_matrix_table = torch.zeros(max_seq_len, d_k, d_k)
        for i in range(max_seq_len):
            block = [self.rotation_block(theta, j, i, d_k) for j in range(d_k//2)]
            rotation_matrix_table[i,:,:] = torch.block_diag(*block)
        return rotation_matrix_table



    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Run RoPE for a given input tensor.
        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor(Query or Key) to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        *prefix_dims, seq_len, d_k = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        rotation_matrix = self.rotation_matrix_table[token_positions]   # (batch_size, seq_len, d_k, d_k)
        x_rotated = rotation_matrix @ x.unsqueeze(-1) # (batch_size, seq_len, d_k, 1)   
        x_rotated = x_rotated.squeeze(-1) # (batch_size, seq_len, d_k)
      
        return x_rotated

