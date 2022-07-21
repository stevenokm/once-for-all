# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse

from ofa.cifar10_classification.data_providers.cifar10 import Cifar10DataProvider
from ofa.cifar10_classification.run_manager import Cifar10RunConfig, RunManager
from ofa.model_zoo import ofa_net_cifar10


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--path",
    help="The path of cifar10",
    type=str,
    default="~/cifar10_data/cifar-10-batches-py",
)
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
parser.add_argument(
    "-b",
    "--batch-size",
    help="The batch on every device for validation",
    type=int,
    default=100,
)
parser.add_argument("-j", "--workers", help="Number of workers", type=int, default=20)
parser.add_argument(
    "-n",
    "--net",
    metavar="OFANET",
    default="ofa_mbv3_d23_e24_k35_w1.0",
    choices=[
        "ofa_mbv3_d23_e24_k35_w1.0",
        "ofa_mbv3_d23_e24_k35_w1.2",
        "ofa_resnet50",
    ],
    help="OFA networks",
)

args = parser.parse_args()
if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
Cifar10DataProvider.DEFAULT_PATH = args.path

ofa_network = ofa_net_cifar10(args.net, pretrained=False)       
ofa_checkpoint_path = "exp/kernel_depth2kernel_depth_width/phase2"
ofa_checkpoint_path = os.path.join(ofa_checkpoint_path, "checkpoint/model_best.pth.tar")
init = torch.load(ofa_checkpoint_path)["state_dict"]
ofa_network.load_state_dict(init)
run_config = Cifar10RunConfig(test_batch_size=args.batch_size, n_worker=args.workers)

ofa_network.set_active_subnet(ks=3, e=2, d=2)
subnet = ofa_network.get_active_subnet(preserve_weight=True)

""" Test sampled subnet 
"""
run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)
run_config.data_provider.assign_active_img_size(32)
run_manager.reset_running_statistics(net=subnet)

print("Test random subnet:")
print(subnet.module_str)

loss, (top1, top5) = run_manager.validate(net=subnet)
print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))
