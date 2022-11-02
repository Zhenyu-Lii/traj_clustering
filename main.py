import argparse
import classification_GCN as GCN
import classification_GAE as GAE

parser = argparse.ArgumentParser()
parser.add_argument('--loss_cuda', default=0, type=int, help='train device')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='epoch size')

if __name__ == "__main__":
    args = parser.parse_args()
    # for k, v in args._get_kwargs():
    #     print("{0} =  {1}".format(k, v))

    # feature learning
    # print("-" * 7 + " start feature learning " + "-" * 7)
    # traj_BERT(args)

    # clustering
    print("-"*7 + " start training " + "-"*7)
    GCN(args)
    # GAE(args)
