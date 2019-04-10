import matplotlib.pyplot as plt
from collections import Counter

def showBatchImage(img, label, out, imgname):
	plt.figure(0, figsize=(15, 15))
	batch_size = min(img.size(0), 5)

	for i in range(batch_size):
		# print(img[0].shape)
		plt.subplot(batch_size, 3, 3*i+1)
		plt.imshow(img[i].cpu().data.numpy().transpose([1, 2, 0]))

		plt.subplot(batch_size, 3, 3*i+2)
		plt.imshow(label[i])

		plt.subplot(batch_size, 3, 3*i+3)
		plt.imshow(out.data.max(1)[1][i])
	plt.show()
	plt.savefig(imgname)


def show_sample_img(img, label):
	plt.figure(0, figsize=(10, 5))

	plt.subplot(1, 2, 1)
	plt.imshow(img.data.numpy().transpose([1, 2, 0]))

	plt.subplot(1, 2, 2)
	plt.imshow(label)

	plt.show()

def showBatchImage_decode(img, label, out, dst, imgname):
	
	plt.figure(0, figsize=(20, 20))
	batch_size = min(img.size(0), 5)

	for i in range(batch_size):
		# print(img[0].shape)
		plt.subplot(batch_size, 3, 3*i+1)
		plt.imshow(img[i].cpu().data.numpy().transpose([1, 2, 0]))

		plt.subplot(batch_size, 3, 3*i+2)
		plt.imshow(dst.decode_segmap(label[i].cpu().data.numpy()))

		plt.subplot(batch_size, 3, 3*i+3)
		plt.imshow(dst.decode_segmap(out.data.max(1)[1][i].cpu().data.numpy()))

	plt.show()
	plt.savefig(imgname)


def getClassWeights(dst):
	weights = Counter()

	for i in range(len(dst)):
	    for m in dst[i][1].unique(): 
	        if weights.get(m.item(), None) == None: 
	            weights[m.item()] = 1
	        else:
	            weights[m.item()] += 1

	for key, val in weights.items(): 
	    weights[key] = len(dst)/val

	return weights 