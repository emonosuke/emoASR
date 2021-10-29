def insert_comment(file_path, comment):
    with open(file_path) as f:
        lines = f.readlines()

    if lines[0] == f"# {comment}\n":
        return

    lines.insert(0, f"# {comment}\n")
    lines.insert(1, "#\n")
    with open(file_path, mode="w") as f:
        f.writelines(lines)


def get_num_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params, num_params_trainable
