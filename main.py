import torch
import os

import argument_parser as parser

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parser.get_args()
    print('running config :', args)