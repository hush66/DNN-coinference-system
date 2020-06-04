from chainer.backends import cuda
import torch

def entropy_gpu(x):
    vec = cuda.elementwise(
        'T x',
        'T y',
        '''
            y = (x == 0) ? 0 : -x*log(x);
        ''',
        'entropy')(x.data)
    return cuda.cupy.sum(vec, 1)


def train(branchyNet, dataloader, num_epoch=20, main=False):
    losses_list = []
    acc_list = []
    epoch_list = range(num_epoch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Starting training...')
    for epoch in epoch_list:
        print("Epoch: %s" % epoch)

        avglosses = []
        avgaccuracies = []

        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)
            if main:
                losses, acc = branchyNet.train_main(imgs, classes)
            else:
                losses, acc = branchyNet.train_branch(imgs, classes)

            if isinstance(losses, list):
                avglosses += losses
            else:
                avglosses.append(losses)
            if isinstance(acc, list):
                avgaccuracies += acc
            else:
                avgaccuracies.append(acc)

        avgloss = sum(avglosses) / len(avglosses)
        avgaccuracy = sum(avgaccuracies) / len(avgaccuracies)

        losses_list.append(avgloss)
        acc_list.append(avgaccuracy)

        print("Losses: %s" % (avgloss))
        print("Acc: %s" % (avgaccuracy))

    return losses_list, acc_list
