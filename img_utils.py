import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import numpy as np 

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
	# plt.show()
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
		plt.imshow(img[i].cpu().detach().data.numpy().transpose([1, 2, 0]))

		plt.subplot(batch_size, 3, 3*i+2)
		plt.imshow(dst.decode_segmap(label[i].cpu().detach().data.numpy()))

		plt.subplot(batch_size, 3, 3*i+3)
		plt.imshow(dst.decode_segmap(out.data.max(1)[1][i].cpu().detach().data.numpy()))

	# plt.show()
	plt.savefig(imgname)
	plt.close()


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

def convert_center_to_lefttop(img, bboxes):
    h, w, _ = img.shape
    bboxes_new = []
    for bbox in bboxes:
        xmin, ymin, wbox, hbox = bbox 
        xmin -= wbox/2
        ymin -= hbox/2
        xmin, ymin, wbox, hbox =  xmin*w, ymin*h , wbox*w, hbox*h
        bboxes_new.append([xmin, ymin, wbox, hbox])
        
    return np.array(bboxes_new)

def convert_center_to_coord(img, bboxes):
    h, w, _ = img.shape
    bboxes_new = []
    for bbox in bboxes:
        xmin, ymin, wbox, hbox = bbox 
        xmin -= wbox/2
        ymin -= hbox/2
        # print(xmin, ymin, wbox, hbox)
        xmin, ymin, wbox, hbox =  xmin*w, ymin*h , wbox*w, hbox*h
        # print(xmin, ymin, wbox, hbox)
        xmax, ymax = xmin + wbox, ymin + hbox
        bboxes_new.append([xmin, ymin, xmax, ymax])
        
    return np.array(bboxes_new)

def merge_cl_bbox(cl ,bbox):
    cl = np.array(list(map(int, cl)))
    cl = np.expand_dims(cl, axis=1)
    print(cl.shape, bbox.shape)
    if bbox.shape[0] == cl.shape[0]:
    	bbox = np.hstack((cl, bbox ))
    return bbox 

def show_bboxes(img, bbox, classes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 10))

    ax1.imshow(img)
    ax2.imshow(img)

    # Create a Rectangle patch
    for obj in bbox:
        cl, xmin, ymin, xmax, ymax = obj
        cl = classes[int(cl)+1]
        print(cl)
        rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,linewidth=2, edgecolor='darkred',facecolor='none')
        ax2.add_patch(rect)
        ax2.text(xmin, ymin, cl, fontsize=15, color ="darkred", fontweight='bold', )

    # Add the patch to the Axes
    plt.show()