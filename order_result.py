import collections


def order_result(file):
    out_dict = dict()
    with open(file, 'r') as f:
        for line in f:
            items = line.split('\t')
            for item in items:
                cell = item.lstrip().rstrip()
                if cell.startswith('all='):
                    out_dict[float(cell[4:])] = line
                    break

    ordered_dict = collections.OrderedDict(sorted(out_dict.items()))
    for _, val in ordered_dict.items():
        print(val)
