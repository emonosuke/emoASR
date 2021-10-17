def insert_comment(file_path, comment):
    with open(file_path) as f:
        lines = f.readlines()

    if lines[0] == f"# {comment}\n":
        return

    lines.insert(0, f"# {comment}\n")
    lines.insert(1, "#\n")
    with open(file_path, mode="w") as f:
        f.writelines(lines)
