import torch.nn as nn
import timm

class DiamondModel(nn.Module):
    """
    A convolutional neural network model for image classification tasks, built upon a ResNet-50 backbone. 
    This model leverages the TIMM library to create the backbone, allowing for easy modification and experimentation 
    with different pre-trained models. The model is designed for binary or multi-class classification tasks.

    Note:
    The model by default does not load pre-trained weights due to the absence of internet access on the cluster. 
    To use a pre-trained model, download the weights manually and adjust the code to load them as needed.

    Attributes:
        backbone (timm.models.resnet.ResNet): The ResNet-50 model used as the backbone for feature extraction.
                                               The model is modified to output the desired number of classes based on 
                                               the `num_classes` parameter.

    Args:
        num_classes (int): The number of classes for the classification task. Default is 2 for binary classification.
    """
    def __init__(self, args):
        super(DiamondModel, self).__init__()
        # Initialize the backbone ResNet-50 model without pre-trained weights
        self.backbone = timm.create_model(args.backbone, pretrained=False, num_classes=args.num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor containing the batch of images. 
                              Tensor dimensions should be `(batch_size, channels, height, width)`.

        Returns:
            torch.Tensor: The output tensor containing the model predictions. The tensor's dimensions will be 
                          `(batch_size, num_classes)`, where `num_classes` is the number provided during model initialization.
        """
        return self.backbone(x)
