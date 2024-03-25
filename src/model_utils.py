from prettytable import PrettyTable


def print_model_summary(model):
    columns = ["Modules", "Parameters", "Param Shape"]
    table = PrettyTable(columns)
    for i, col in enumerate(columns):
        if i == 0:
            table.align[col] = "l"
        else:
            table.align[col] = "r"
    total_param_nums = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_nums = parameter.numel()
        param_shape = list(parameter.shape)
        table.add_row([name, "{:,}".format(param_nums), "{}".format(param_shape)])
        total_param_nums += param_nums

    separator = ["-" * len(x) for x in table.field_names]
    table.add_row(separator)
    table.add_row(["Total", "{:,}".format(total_param_nums), "{}".format("_")])

    print(table, "\n")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item
