import cv2
from utils.data import create_dataloader

# hard code, path to list image file
train_path = '/home/heligate/Documents/ScaledYOLOv4/coco/train2017.txt'

# create data loader
dataloader = create_dataloader(train_path, batch_size=8, img_size=640)

# index sample in batch w
batch_index = 7
target_batch = 150

for i, item in enumerate(iter(dataloader)):
       data = item
       if i == target_batch:
              break

img, label = data
img = img[batch_index].permute(1, 2, 0).numpy()
labels = label.numpy()

print('label')
print(label)

cv2.imwrite('img.jpg', img)
img = cv2.imread('img.jpg')

for i in range(labels.shape[0]):
       if label[i, 0] != batch_index:
              continue
       xywh = labels[i, -4:]
       x1, y1, x2, y2 = xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
       x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])

       img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

cv2.imwrite('img.jpg', img)