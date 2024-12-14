import torch


my_dict = dict(a = 1, b = 2)

my_dict['a'] = my_dict.get('a', 0)
print(my_dict)