import torch.nn.functional as F
import torch
import eval_metrics as ev

def calc_metrics(label, out_pred):
    n = label.size(0)
    # print("Batch Size: ", n)
    pix_acc = 0
    mean_acc = 0
    mean_iou = 0
    freq_w_iou = 0

    for i in range(n):
        gt = label[i].cpu().data.numpy()
        pred = out_pred[i].cpu().data.numpy()
        
        pix_acc += ev.pixel_accuracy(gt, pred)
        mean_acc += ev.mean_accuracy(gt, pred)
        mean_iou += ev.mean_IU(gt, pred)
        freq_w_iou += ev.frequency_weighted_IU(gt, pred)

    # print("Pixel Accuracy: ", pix_acc/n)
    # print("Mean Accuracy: ", mean_acc/n)
    # print("Mean IOU: ", mean_iou/n)
    # print("Frequency Weighted IOU: ", freq_w_iou/n)

    return pix_acc/n, mean_acc/n, mean_iou/n, freq_w_iou/n

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    print(target.size())
    nt, c, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsqueeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.squeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(input, target, weight=weight, size_average=size_average, ignore_index=250)

    return loss

def dice_loss(input ,target):
    smooth = 1.
    # have to use contiguous since they may from a torch.view op
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    # return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
    return 1 - ((intersection + smooth) / (A_sum + B_sum - intersection + smooth))


# def dice_loss(input, target, num_classes=21):
#     smooth = 1.
#     # inp = input[:, 1:]
#     iflat = input[:, 1:, :, :].contiguous().view(-1)
#     tflat = target[:, 1:, :, :].contiguous().view(-1)
#     intersection = (iflat * tflat).sum()
    
#     foreground = 1 - ((2. * intersection + smooth) /
#               (iflat.sum() + tflat.sum() + smooth))
    
#     iflat_back = input[:, 0, :, :].contiguous().view(-1)
#     tflat_back = target[:, 0, :, :].contiguous().view(-1)
#     intersection_back = (iflat_back * tflat_back).sum()
    
#     background = 1 - ((2. * intersection_back + smooth) /
#               (iflat_back.sum() + tflat_back.sum() + smooth))
    
#     return foreground*0.7 + background*0.3


def iou(pred, target, num_classes=21):
	
	smooth = 1.0
	# have to use contiguous since they may from a torch.view op
	iflat = pred.contiguous().view(-1)
	tflat = target.contiguous().view(-1)
	intersection = (iflat * tflat).sum()

	A_sum = torch.sum(tflat * iflat)
	B_sum = torch.sum(tflat * tflat)

	return ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def accuracy(pred, gt):
    return torch.mean((pred == gt).float())