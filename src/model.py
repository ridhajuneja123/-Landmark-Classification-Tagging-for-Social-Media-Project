import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

#         super().__init__()

#         # YOUR CODE HERE
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))
#         self.cnn_layer0 = nn.Conv2d(3,32,kernel_size=3,padding='same')
        
#         self.cnn_layer1 = nn.Conv2d(32,64,kernel_size=3,padding='same')
#         self.max_pooling1 = nn.MaxPool2d(2,2)
        
#         self.cnn_layer2 = nn.Conv2d(64,128,kernel_size=3,padding='same')
#         self.max_pooling2 =  nn.MaxPool2d(2,2)
        
#         self.cnn_layer3 = nn.Conv2d(128,256,kernel_size=3,padding='same')
#         self.max_pooling3 =  nn.MaxPool2d(2,2)
        
        
#         self.cnn_layer4 = nn.Conv2d(256,512,kernel_size=3,padding='same')
#         self.max_pooling4 =  nn.MaxPool2d(2,2)
        
#         self.cnn_layer5 = nn.Conv2d(512,512,kernel_size=3,padding='same')
#         self.max_pooling5 =  nn.MaxPool2d(2,2)
        
#         self.flatten = nn.Flatten(start_dim=1)
#         self.droupt1 = nn.Dropout(0.25)
#         self.fc1 = nn.Linear(25088,6272)
#         self.fc2 = nn.Linear(6272,num_classes)
        

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)
#         x = self.cnn_layer0(x)
# #         print(x.size())
#         x = self.cnn_layer1(x)
#         x = self.max_pooling1(x)
#         x = self.cnn_layer2(x)
#         x = self.max_pooling2(x)
#         x = F.relu(x)
#         x = self.cnn_layer3(x)
#         x = self.max_pooling3(x)
#         x = F.relu(x)
#         x = self.cnn_layer4(x)
#         x = self.max_pooling4(x)
#         x = F.relu(x)
#         x = self.cnn_layer5(x)
#         x = self.max_pooling5(x)
#         x = F.relu(x)
#         x = self.flatten(x)
#         x = self.droupt1(x)
#         x = self.fc1(x)
#         x = x = F.relu(x)
#         x = self.fc2(x)
        
        
#         return x

# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

#         super().__init__()

#         # YOUR CODE HERE
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))
        
#         self.cnn_layer1 = nn.Conv2d(3,32,kernel_size=5,padding='same')
#         self.max_pooling1 = nn.MaxPool2d(4,4)
        
#         self.cnn_layer2 = nn.Conv2d(32,64,kernel_size=5,padding='same')
#         self.max_pooling2 =  nn.MaxPool2d(2,2)
        
#         self.cnn_layer3 = nn.Conv2d(64,128,kernel_size=3,padding='same')
#         self.max_pooling3 =  nn.MaxPool2d(2,2)
        
#         self.cnn_layer4 = nn.Conv2d(128,256,kernel_size=3,padding='same')
#         self.max_pooling4 =  nn.MaxPool2d(2,2)
        
       
        
#         self.flatten = nn.Flatten(start_dim=1)
#         self.droupt1 = nn.Dropout(0.25)
#         self.fc1 = nn.Linear(12544,3136)
#         self.fc2 = nn.Linear(3136,num_classes)
        

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)
# #         print(x.size())

#         x = self.cnn_layer1(x)
#         x = self.max_pooling1(x)
        
#         x = self.cnn_layer2(x)
#         x = self.max_pooling2(x)
#         x = F.relu(x)
        
#         x = self.cnn_layer3(x)
#         x = self.max_pooling3(x)
#         x = F.relu(x)
        
#         x = self.cnn_layer4(x)
#         x = self.max_pooling4(x)
#         x = F.relu(x)
       
#         x = self.flatten(x)
#         x = self.droupt1(x)
#         x = self.fc1(x)
#         x = x = F.relu(x)
#         x = self.fc2(x)
        
#         return x
    
    
    
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super(MyModel, self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
#         nn.Conv2d,
#         nn.ReLU(),
#         nn.BatchNorm2d,
#         nn.MaxPool2d,
        
        self.cnn_layer1 =  nn.Conv2d(3, 32, 3, padding=1)
        self.cnn_layer2 =  nn.Conv2d(32, 64, 3, padding=1)
        self.cnn_layer3 =  nn.Conv2d(64, 128, 3, padding=1)
        self.cnn_layer4 =  nn.Conv2d(128, 256, 3, padding=1)
        self.cnn_layer5 =  nn.Conv2d(256, 512, 3, padding=1)
        
        
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.batchnorm5 = nn.BatchNorm2d(512)
       
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
       
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.maxpool5 = nn.MaxPool2d(2, 2)
       
        self.flatten =  nn.Flatten(start_dim=1)
        self.linear1 =  nn.Linear(512 * 7 * 7, 1024)
        self.batchnorm6 =nn.BatchNorm1d(1024)
        self.linear2 =  nn.Linear(1024, 512)
        self.linear3 =  nn.Linear(512, num_classes)
       
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.cnn_layer1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        
        x = self.cnn_layer2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        
        x = self.cnn_layer3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        
        
        x = self.cnn_layer4(x)
        x = self.relu4(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)
        
        
        x = self.cnn_layer5(x)
        x = self.relu5(x)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)
        
        
       
        x = self.flatten(x)
        
        
        x = F.relu(self.linear1(x))
        x=  self.batchnorm6(x)
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        
        
        
        
        return x



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
