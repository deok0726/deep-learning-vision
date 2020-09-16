import argparse

parser = argparse.ArgumentParser(description='anomaly_detection')

# main
parser.add_argument('--backbone_name', type=str, default='AE', help="model name")

# data loader

# train

# save

def get_args():
    args = parser.parse_args()
    return args