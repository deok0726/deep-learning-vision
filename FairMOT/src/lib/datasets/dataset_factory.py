from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset


def get_dataset(dataset, task):
  if task == 'mot':
    print(dataset)
    return JointDataset
  else:
    return None
  
