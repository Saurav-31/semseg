dataset_path = "/mnt/sqnap1/saugupt/public_datasets/PascalVoc2012/"
optimizer_config = {
    'lr': 1.0e-08,
    'weight_decay': 0.0008,
    'momentum': 0.99
}

loss_params = {
    'size_average': True
}

START_EPOCH = 0
NUM_EPOCHS = 50000
EXPERIMENT_NAME = "mantis_5back_foreground_big_contd"
CONTINUE_TRAINING = True

save_path = "./data/{}/".format(EXPERIMENT_NAME)

os.makedirs(save_path, exist_ok=True)

data_voc = PascalVOCLoader(dataset_path, is_transform=True, img_size=256)
# data_voc = PascalVOCLoader(dataset_path, is_transform=True, img_size=256, color_path="./data/color_quant.npy")
trainLoader = data.DataLoader(data_voc, batch_size=64, num_workers=1, shuffle=True)

if CONTINUE_TRAINING:
	print("Loading saved model to continue training...")
	net = torch.load("./data/mantis_5back_foreground_big/model_epoch_2000.pth")
else:
	net = MSD5Net(num_layers=64, in_channels=3, out_channels=21)

net.cuda()

train(trainLoader, net, optimizer_config, loss_params, start_epoch=START_EPOCH, num_epochs=NUM_EPOCHS, save_path=save_path)



