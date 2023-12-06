import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("MPS device not found.")
print('Device:', device)

data_dir='../datasets/miniImageNet/'
image_size=84
batch_size=128
num_workers=4

# dataset = torchvision.datasets.ImageFolder(
#     root=data_dir,
#     # transform=transforms.Compose([
#     #     transforms.Resize(image_size),
#     #     transforms.CenterCrop(image_size),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     # ])
# )
#
# # Create the dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
# # Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(5, 5))
# plt.axis('off')
# plt.title('The Training Dataset')
# plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:36], padding=2, normalize=True, nrow=6).cpu(),(1,2,0)))

from src.mini_imagenet_dataloader import MiniImageNetDataLoader

dataloader = MiniImageNetDataLoader(shot_num=5, way_num=5, episode_test_sample_num=15)

dataloader.generate_data_list(phase='train')
dataloader.generate_data_list(phase='val')
dataloader.generate_data_list(phase='test')

dataloader.load_list(phase='all')

print(len(dataloader.val_labels))

