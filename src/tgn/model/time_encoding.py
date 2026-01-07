import torch


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super().__init__()

    self.dimension = dimension


    # [dimension]
    steps = torch.linspace(0, 9, dimension)
    freq = 1.0/ (10.0 ** steps)

    # [dimension]
    self.w = torch.nn.Parameter(freq.float())

    # [dimension]
    self.bias = torch.nn.Parameter(torch.zeros(dimension).float())


  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    # [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=-1)

    # t: [batch_size, seq_len, 1] -> broadcasting [batch_size, seq_len, dimension]
    # w: [dimension]
    # b: [dimension]

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(t * self.w + self.bias)

    return output