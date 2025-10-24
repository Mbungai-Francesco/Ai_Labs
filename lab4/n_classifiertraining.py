import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # to avoid crash on some systems
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Function to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images and check the size
    for images,labels in trainloader:
        print('batch size:', images.size(0))
        print('color channels :', images.size(1))
        print('Image size:'+ str(images.size(2))+ 'x'+ str(images.size(3)))
        break # we just want to fetch the first batch

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s\t' % classes[labels[j]] for j in range(4)))

    class ImageNet(nn.Module):
        def __init__(self):
            super(ImageNet, self).__init__()
            ## Convolutional layers, where weights represent conv kernels
            # 1 input image channel, 6 output channels, 3x3 square convolution
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
            # 6 input channels (the output of the last layer), 16 output channels, 3x3 square convolution
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

            ## Linear layer: MLP, i.e. fully-connected layer.
            self.fc1 = nn.Linear(in_features = 16 * 6 * 6, out_features = 120)  # 6*6 from the image dimension, and 16 for the number of channels
            self.fc2 = nn.Linear(in_features = 120, out_features = 84) # 120 is output of the previous layer.
            self.fc3 = nn.Linear(in_features = 84, out_features = 10) # 84 is the output of the previous layer, 10 is the number of classes.

        def forward(self, x):
            # Conv1, then max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # Conv2, then max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
            x = x.view(-1, self.num_flat_features(x)) # Reshape each image, processed by conv, into a vector (required for linear layers)
            # 1st Linear layer
            x = F.relu(self.fc1(x))
            # 2nd Linear layer
            x = F.relu(self.fc2(x))
            # 3rd Linear layer
            x = self.fc3(x)
            return x

        def num_flat_features(self, x): # Computes the number of flat (*"vectorized"*) features from a 2D conv.
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    net = ImageNet()
    print(net)
    
     # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    n_epochs=2
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[epoch %d, batch %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    
    # Test the network on the test data
    for images,labels in testloader:
        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(images.shape[0])))
        break # here again we just want to fetch the first batch

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(images.shape[0])))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(images.shape[0])))


    correct = 0
    total = 0
    # torch.no_grad is important for TESTING.  
    with torch.no_grad(): # This line actually disables the gradient computation.
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


    ## the first batch is the "images" tensor
    print(f"Tensor of the first batch, shape : {images.shape} ")

    ## We use the first layer of the model to process the input
    processed = net.conv1(images)

    ## We keep only one image of the batch to visualize it
    index_img = 7 ## this is between 0 and batch_size - 1 

    image = images[index_img]
    image = image.unsqueeze(0) ## this is needed to add a singleton dimension to the tensor, so that we can visualize it with make_grid
    # here we will add a singleton dimension as if we had a batch of size 1 to visualize, but we are keeping the three channels to keep the colors 


    ## same thing with the output of the first convolutional layer
    processed = processed[index_img]
    processed = processed.unsqueeze(1) ## this is needed to add a singleton dimension to the tensor, so that we can visualize it with make_grid
    # here, remember that we want to visualize the output of the first convolutional layer, which has 6 channels. We need to add a singleton dimension to the tensor to visualize it with make_grid
    # we keep the six feature maps as the "batch size" of the tensor, and we add a singleton dimension as a single channel

    # visualize
    print("Original image")
    imshow(torchvision.utils.make_grid(image.detach()))

    print("Outputs of the first convolutional layer")
    imshow(torchvision.utils.make_grid(processed.detach(),scale_each=True, normalize=True))


    # Effect of relu and max pooling
    processed_relumaxpool = F.max_pool2d(F.relu(net.conv1(images)), (2, 2))

    ## same thing with the output of the first convolutional layer
    processed_relumaxpool = processed_relumaxpool[index_img]
    processed_relumaxpool = processed_relumaxpool.unsqueeze(1)
    # visualize
    print("Original image")
    imshow(torchvision.utils.make_grid(image.detach()))

    print("Outputs of the first convolutional layer")
    imshow(torchvision.utils.make_grid(processed.detach(),normalize=True,value_range=(-1,1),padding=0))

    ## print ranges 
    print(f"Range of the original image : [{torch.min(image).item()}, {torch.max(image).item()}]")
    print(f"Range of the output of the first convolutional layer : [{torch.min(processed).item()}, {torch.max(processed).item()}]")
    print(f"Range of the output of the first convolutional layer after Relu and Max Pool : [{torch.min(processed_relumaxpool).item()}, {torch.max(processed_relumaxpool).item()}]")


    print("After Relu and Max Pool")
    imshow(torchvision.utils.make_grid(processed_relumaxpool.detach(), normalize=True,value_range=(-1,1),padding=0))
        
    
    PATH = './cifar_net.pth'
    # Save the model
    torch.save(net.state_dict(), PATH)

    # Load a saved model
    net = ImageNet()
    net.load_state_dict(torch.load(PATH))

