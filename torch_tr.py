import torch

size = 3
a = torch.ones(1, size, size)

# Extract the upper triangular part
upper_triangular = torch.triu(a, diagonal=1).type(torch.int)

print("Original Tensor:")
print(a)
print("\nUpper Triangular Part:")
print(upper_triangular == 0)