from utils.data import create_dataloader


train_path = '/home/heligate/Documents/ScaledYOLOv4/coco/train2017.txt'

dataloader = create_dataloader(train_path, batch_size=8, img_size=640)

# Test
data = next(iter(dataloader))
print(data)

for images, labels in data:
    print('images: ')
    print(images)
    # print('labels: ')
    # print(labels)