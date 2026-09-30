import torch
import torch.utils.data
import math


# define data loader
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        # read file to get string array by line
        with open(data_file, "r") as file:
            lines = [line.strip() for line in file]
        self.data = lines
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].split(":")
        input = torch.zeros(len(line[0]))
        for i in range(0, len(line[0])):
            input[i] = int(line[0][i])

        output = torch.zeros(len(line[1]))
        for i in range(0, len(line[1])):
            output[i] = int(line[1][i])

        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            output = self.target_transform(output)

        return input, output


# This dataset loads the data in shape of an image -> channel=1,width=8,height=4 -> [1, 4, 8] !Pytorch images are (channel, height, width)!
class MyConvDataset(torch.utils.data.Dataset):
    def __init__(self, data_file,width, height, transform=None, target_transform=None):
        # read file to get string array by line
        with open(data_file, "r") as file:
            lines = [line.strip() for line in file]
        self.data = lines
        self.width = width
        self.height = height
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        width = self.width
        height = self.height
        line = self.data[idx].split(":")
        input = torch.zeros(1, height, width)
        for row in range(0, height):
            for col in range(0, width):
                input[0, row, col] = int(line[0][row * width + col])

        output = torch.zeros(width * height)
        for i in range(0, width * height):
            output[i] = int(line[1][i])

        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            output = self.target_transform(output)

        return input, output

# This dataset loads the data in shape of an image -> channel=1,width=8,height=4 -> [1, 4, 8] !Pytorch images are (channel, height, width)!
class MyFootPositionRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_file,width, height, transform=None, target_transform=None):
        # read file to get string array by line
        with open(data_file, "r") as file:
            lines = [line.strip() for line in file]

        dInput = []
        dLabel = []
        for idx in range(0, len(lines)):
            line = lines[idx].split(":")
            input = torch.zeros(1, height, width)
            for row in range(0, height):
                for col in range(0, width):
                    input[0, row, col] = int(line[0][row * width + col])

            footCoordsArr = []
            footCoordsStrings = line[1].split(" ")
            for i in range(len(footCoordsStrings)):
                curr = footCoordsStrings[i]
                if not curr =="":
                    x = curr.split(",")[0]
                    y = curr.split(",")[1]
                    footCoordsArr.append([x,y])

            output = generateGtTensorFromFootCoordinates(footCoordsArr, s = 10, b = 2)

            if transform:
                input = transform(input)
            if target_transform:
                output = target_transform(output)
            dInput.append(input)
            dLabel.append(output)

        self.dataInput = dInput
        self.dataLabel = dLabel

    def __len__(self):
        return len(self.dataInput)

    def __getitem__(self, idx):
        return self.dataInput[idx], self.dataLabel[idx]

def generateGtTensorFromFootCoordinates(footCoordsArr, s = 10, b = 2)-> torch.Tensor:
    output_space_size = 500 # the output space is 500cm x 500cm
    temp = [[[0.0 for _ in range(3)] for _ in range(s)] for _ in range(s)]
    for x_pos, y_pos in footCoordsArr:
        px_per_cell = output_space_size/s
        x_idx = math.floor(int(x_pos) / px_per_cell)
        y_idx = math.floor(int(y_pos) / px_per_cell)

        # convert output to be relative to the cell position and size
        cell_x = x_idx * px_per_cell
        cell_y = y_idx * px_per_cell
        relative_x_pos = (int(x_pos) - cell_x) / px_per_cell
        relative_y_pos = (int(y_pos) - cell_y) / px_per_cell
        temp[x_idx][y_idx][0]= float(relative_x_pos)
        temp[x_idx][y_idx][1]= float(relative_y_pos)
        temp[x_idx][y_idx][2]= 1.0
        #print("x_idx: ",x_idx,", y_idx: ",y_idx, ", cell_x: ", cell_x, ", cell_y: ", cell_y,", x_pos: ",x_pos,", y_pos: ",y_pos,", rel_x: " , relative_x_pos, ", rel_y: ", relative_y_pos)


    # target tensor is of size s x s x 3 -> there is no b here because there can only be one object for each cell
    tensor = torch.zeros( s, s, 3)
    for x_idx in range(s):
        for y_idx in range(s):
                for k in range(3):
                    tensor[x_idx,y_idx,k] = temp[x_idx][y_idx][k]

    return tensor

# this dataset is for usage with PixelEmbeddingLoss
class MyInstanceSegDataset(torch.utils.data.Dataset):
    def __init__(self, data_file,width,height, transform=None, target_transform=None):
        # read file to get string array by line
        with open(data_file, "r") as file:
            lines = [line.strip() for line in file]

        dInput = []
        dLabel = []
        for idx in range(0, len(lines)):
            line = lines[idx].split(":")
            input = torch.zeros(1, height, width)
            for row in range(0, height):
                for col in range(0, width):
                    input[0, row, col] = int(line[0][row * width + col])

            instanceMasks = line[1].split(",")
            instanceMasks.pop() # pop because of "," at the end the last element was empty
            """
            output = torch.zeros(height* width,len(instanceMasks))
            for pixel_pos in range(0, height*width):
                    for i in range(0, len(instanceMasks)):
                        output[pixel_pos, i] = int(instanceMasks[i][pixel_pos])
"""
            output = torch.zeros(len(instanceMasks),height, width)
            for h in range(0,height):
                for w in range(0,width):
                    for i in range(0, len(instanceMasks)):
                        output[i,h,w] = int(instanceMasks[i][h*width+w])

            if transform:
                input = transform(input)
            if target_transform:
                output = target_transform(output)
            dInput.append(input)
            dLabel.append(output)

        self.dataInput = dInput
        self.dataLabel = dLabel

    def __len__(self):
        return len(self.dataInput)

    def __getitem__(self, idx):
        return self.dataInput[idx], self.dataLabel[idx]

def collate_fn(data):
    images, labels = zip(*data)
    images = torch.stack(images,dim=0)
    max_size = tuple(max(s) for s in zip(*[label.shape for label in labels]))
    padded_labels = torch.zeros((len(labels),*max_size), dtype=labels[0].dtype)
    for i, label in enumerate(labels):
        padded_labels[i, :label.shape[0], :label.shape[1]] =label 
    return images, padded_labels

# this dataset is for usage with PixelEmbeddingLoss
class MySemanticSegDataset(torch.utils.data.Dataset):
    def __init__(self, data_file,width,height, transform=None, target_transform=None):
        # read file to get string array by line
        with open(data_file, "r") as file:
            lines = [line.strip() for line in file]

        dInput = []
        dLabel = []
        for idx in range(0, len(lines)):
            line = lines[idx].split(":")
            input = torch.zeros(1, height, width)
            for row in range(0, height):
                for col in range(0, width):
                    input[0, row, col] = int(line[0][row * width + col])

            semanticMasks= line[1].split(",")
            semanticMasks.pop() # pop because of "," at the end the last element was empty
            output = torch.zeros(len(semanticMasks),height, width)
            for h in range(0,height):
                for w in range(0,width):
                    for i in range(0, len(semanticMasks)):
                        output[i,h,w] = int(semanticMasks[i][h*width+w])

            if transform:
                input = transform(input)
            if target_transform:
                output = target_transform(output)
            dInput.append(input)
            dLabel.append(output)

        self.dataInput = dInput
        self.dataLabel = dLabel

    def __len__(self):
        return len(self.dataInput)

    def __getitem__(self, idx):
        return self.dataInput[idx], self.dataLabel[idx]

