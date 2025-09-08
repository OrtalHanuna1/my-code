import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt

from torchvision import transforms
from config import (
    VIS_LABEL_MAP as viz_map
)

plt.style.use('ggplot')

def set_class_values(all_classes, classes_to_train):
    """
    This (`class_values`) assigns a specific class label to the each of the classes.
    For example, `animal=0`, `archway=1`, and so on.

    :param all_classes: List containing all class names.
    :param classes_to_train: List containing class names to train.
    """
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """
    Encode pixels to contiguous class ids 0..len(class_values)-1 for the selected subset.
    Pixels of classes not in the subset are mapped to 0 (background).
    """
    h, w = mask.shape[:2]
    label_mask = np.zeros((h, w), dtype=np.uint8)

    # map from original VOC class id -> new contiguous id
    value_to_new = {v: i for i, v in enumerate(class_values)}

    mask_rgb = mask.astype(np.uint8)
    for orig_idx, color in enumerate(label_colors_list):
        color_arr = np.array(color, dtype=np.uint8)
        matches = np.all(mask_rgb == color_arr, axis=-1)
        if orig_idx in value_to_new:
            label_mask[matches] = value_to_new[orig_idx]
        else:
            # map unselected classes to background
            label_mask[matches] = 0

    return label_mask.astype(int)

def draw_translucent_seg_maps(
    data, 
    output, 
    epoch, 
    i, 
    val_seg_dir, 
    label_colors_list,
    writer=None,
    tb_tag='Images/val_overlay',
    target=None,
):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """
    alpha = 1 # how much transparency
    beta = 0.8 # alpha + beta should be 1
    gamma = 0 # contrast

    seg_map = output[0] # use only one output from the batch
    seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()

    # keep original input for TB
    image_chw = np.array(data[0].cpu())  # C,H,W in [0,1]
    # for overlay (work in H,W,C and 0..255 BGR)
    image = np.transpose(image_chw, (1, 2, 0)).astype(np.float32) * 255.0

    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)

    for label_num in range(0, len(label_colors_list)):
        index = seg_map == label_num
        red_map[index] = np.array(viz_map)[label_num, 0]
        green_map[index] = np.array(viz_map)[label_num, 1]
        blue_map[index] = np.array(viz_map)[label_num, 2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2).astype(np.float32)
    # convert color to BGR format for OpenCV
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image_bgr, alpha, rgb_bgr, beta, gamma, image_bgr)
    cv2.imwrite(f"{val_seg_dir}/e{epoch}_b{i}.jpg", image_bgr)

    # TensorBoard images: input, pred mask, GT mask (if given), overlay
    if writer is not None:
        # input
        inp_tensor = torch.from_numpy(image_chw).float()  # C,H,W in [0,1]
        writer.add_image('Images/val_input', inp_tensor, global_step=epoch)

        # predicted mask (RGB)
        pred_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
        pred_tensor = torch.from_numpy(pred_rgb).permute(2, 0, 1).float() / 255.0
        writer.add_image('Images/val_pred_mask', pred_tensor, global_step=epoch)

        # ground-truth mask (RGB) if available
        if target is not None:
            gt = np.array(target[0].detach().cpu(), dtype=np.int32)
            r = np.zeros_like(gt, dtype=np.uint8)
            g = np.zeros_like(gt, dtype=np.uint8)
            b = np.zeros_like(gt, dtype=np.uint8)
            for label_num in range(0, len(label_colors_list)):
                idx = gt == label_num
                r[idx] = np.array(viz_map)[label_num, 0]
                g[idx] = np.array(viz_map)[label_num, 1]
                b[idx] = np.array(viz_map)[label_num, 2]
            gt_rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
            gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).float() / 255.0
            writer.add_image('Images/val_gt_mask', gt_tensor, global_step=epoch)

        # overlay (RGB)
        overlay_rgb = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        overlay_tensor = torch.from_numpy(overlay_rgb).permute(2, 0, 1).float() / 255.0
        writer.add_image(tb_tag, overlay_tensor, global_step=epoch)

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, epoch, model, out_dir, name='model'
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))

class SaveBestModelIOU:
    """
    Class to save the best model while training. If the current epoch's 
    IoU is higher than the previous highest, then save the
    model state.
    """
    def __init__(self, best_iou=float(0)):
        self.best_iou = best_iou
        
    def __call__(self, current_iou, epoch, model, out_dir, name='model'):
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            print(f"\nBest validation IoU: {self.best_iou}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))

def save_model(epochs, model, optimizer, criterion, out_dir, name='model'):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(out_dir, name+'.pth'))

def save_plots(
    train_acc, valid_acc, 
    train_loss, valid_loss, 
    train_miou, valid_miou, 
    out_dir
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    # mIOU plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_miou, color='tab:blue', linestyle='-', 
        label='train mIoU'
    )
    plt.plot(
        valid_miou, color='tab:red', linestyle='-', 
        label='validataion mIoU'
    )
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'miou.png'))

def get_segment_labels(image, model, device):
    image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
    image = image.unsqueeze(0) # add a batch dimension
    with torch.no_grad():
        outputs = model(image.to(device))
    return outputs

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(viz_map)):
        index = labels == label_num
        red_map[index] = np.array(viz_map)[label_num, 0]
        green_map[index] = np.array(viz_map)[label_num, 1]
        blue_map[index] = np.array(viz_map)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image, segmented_image):
    alpha = 1 # transparency for the original image
    beta = 1.0 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image