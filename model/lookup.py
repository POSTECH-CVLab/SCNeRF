import torch


def lookup_xy(L, k, loc, rank, level=8):
    
    candidate = torch.arange(0, L+1, L / 2**level).to(rank)
    d = (candidate - L / 2) / (L / 2)
    val = (1 + k[0] * d ** 2 + k[1] * d ** 4) * (candidate - L / 2) + L / 2

    location = torch.searchsorted(val, loc.contiguous())
    
    valid_loc = torch.logical_and(location <= 2 ** level, location > 0)

    location[location <= 0] = 1
    location[location > 2 ** level] = 2 ** level

    return valid_loc, location, val, candidate


def lookup(W, H, k, x, y, rank, level=8):

    valid_loc_x, loc_x, val_x, cand_x = lookup_xy(W, k, x, rank=rank, level=level)
    valid_loc_y, loc_y, val_y, cand_y = lookup_xy(H, k, y, rank=rank, level=level)
    
    valid_pos = torch.logical_and(valid_loc_x, valid_loc_y)

    inter_x, inter_y = val_x[loc_x] - val_x[loc_x - 1], val_y[loc_y] - val_y[loc_y - 1]
    x_cand = cand_x[loc_x] * (x - val_x[loc_x - 1]) + cand_x[loc_x - 1] * (val_x[loc_x] - x)
    y_cand = cand_y[loc_y] * (y - val_y[loc_y - 1]) + cand_y[loc_y - 1] * (val_y[loc_y] - y)

    x_cand, y_cand = x_cand / inter_x, y_cand / inter_y

    return valid_pos, torch.stack([x_cand, y_cand]).T
    