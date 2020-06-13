from networks import alex_cifar10
from datasets import pcifar10
from branchynet import utils
from config import *

if __name__ == '__main__':
	#Get the B-AlexNet architecture to classify images using Cifar10 dataset
	branchyNet = alex_cifar10.get_network()

	dataloader = pcifar10.get_data()

	branchyNet.to_gpu()

	branchyNet.training()

	# Train only the main branch of BranchyNet.
	main_loss, main_acc = utils.train(branchyNet, dataloader, main=True, batchsize=TRAIN_BATCHSIZE,
		num_epoch=MAIN_TRAIN_NUM_EPOCHS)

	# Train the side branches of BranchyNet.
	main_loss, main_acc = utils.train(branchyNet,dataloader, batchsize=TRAIN_BATCHSIZE,
		num_epoch=BRANCH_TRAIN_NUM_EPOCHS)

	branchyNet.save_branchyNet('trained_model/BranchyAlexNet(100,200).pt')