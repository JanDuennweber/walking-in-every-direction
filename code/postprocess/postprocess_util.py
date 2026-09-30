import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from .mean_shift_clustering import \
    apply_mean_shift_to_model_output_without_background_pixels


def post_process_model_output(model_output, semantic_mask, bandwidth= 2.5):
    # clustered_output = apply_mean_shift_to_model_output(model_output,bandwidth=2.5) # the model was trained with delta_var = 0.5
    clustered_output = apply_mean_shift_to_model_output_without_background_pixels(
        model_output, semantic_mask, bandwidth=bandwidth
    )

    # multiply the clusterd_output with a semantic segmentation mask to filter out background pixels -> in this case the input happens to be a perfect sementic segmentation mask differentiating between foreground and background
    # clusterd_output: bs x height x width
    # input_with_bs: bs x channels x height x width (where channels is alway 1)
    # semantic_mask_squeezed: bs x height x width
    semantic_mask_squeezed = torch.squeeze(semantic_mask, dim=1)
    filtered = (clustered_output + 1) * semantic_mask_squeezed.detach().numpy()

    return filtered


def calculate_iou(pred, label):
    intersection = np.logical_and(pred, label).sum()
    union = np.logical_or(pred, label).sum()
    if union == 0:
        return 0
    return intersection / union


def calculate_precision_for_batch(
    processed_model_output: np.ndarray, labels_pytorch: torch.Tensor, iou_threshold=0.5
):
    # processed_model_output: bs x height x width
    # label: bs x numClusters x height x width
    bs, height, width = processed_model_output.shape
    labels = labels_pytorch.detach().numpy()

    true_positives = 0
    false_positives = 0
    total_instances = 0
    for i in range(bs):
        pred = processed_model_output[i]
        label = labels[i]
        total_instances += label.shape[0]

        false_pos_indices = set()
        false_pos_iou = []
        false_pos_label = set()
        for j in range(label.shape[0]):  # Iterate over numClusters
            label_instance = label[j]
            label_instance_bool = label_instance == 1
            # Find unique instances in the prediction and remove the background (0)
            unique_instances = np.setdiff1d(np.unique(pred), [0])
            best_iou = 0
            for instance in unique_instances:
                pred_instance = pred == instance
                iou = calculate_iou(pred_instance, label_instance_bool)
                best_iou = max(best_iou, iou)

            if best_iou > iou_threshold:
                true_positives += 1
            else:
                false_positives += 1
                false_pos_indices.add(i)
                false_pos_iou.append(best_iou)
                false_pos_label.add(convertLabelToString(label))

        if len(false_pos_indices) >= 1:
            print(
                "False Positive. bs: ", false_pos_indices, " iou: ", false_pos_iou
            )  # " labels: ", false_pos_label

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_instances

    return precision, recall


def calculate_metrics_for_batch(
    processed_model_output: np.ndarray, labels_pytorch: torch.Tensor, iou_threshold=0.5
):
    # processed_model_output: bs x height x width
    # label: bs x numClusters x height x width
    bs, height, width = processed_model_output.shape
    labels = labels_pytorch.detach().numpy()

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    total_instances = 0
    for i in range(bs):
        pred = processed_model_output[i]
        label = labels[i]
        total_instances += label.shape[0]

        false_pos_indices = set()
        false_pos_iou = []
        false_pos_label = set()
        for j in range(label.shape[0]):  # Iterate over numClusters
            label_instance = label[j]
            label_instance_bool = label_instance == 1
            # Find unique instances in the prediction and remove the background (0)
            unique_instances = np.setdiff1d(np.unique(pred), [0])
            best_iou = 0
            for instance in unique_instances:
                pred_instance = pred == instance
                iou = calculate_iou(pred_instance, label_instance_bool)
                best_iou = max(best_iou, iou)

            if best_iou > iou_threshold:
                true_positives += 1
            else:
                false_positives += 1
                false_pos_indices.add(i)
                false_pos_iou.append(best_iou)
                false_pos_label.add(convertLabelToString(label))

        if len(false_pos_indices) >= 1:
            print(
                "False Positive. bs: ", false_pos_indices, " iou: ", false_pos_iou
            )  # " labels: ", false_pos_label

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_instances
    accuracy = 0

    return precision, recall, accuracy

def calculate_pixel_accuracy_for_batch_semantic_seg(
    model_output_pytorch: torch.Tensor, labels_pytorch: torch.Tensor
, threshold = 0.5):
    # model_ouptut_pytorch: bs x 1 x height x width
    # label: bs x height x width
    bs,c, height, width = model_output_pytorch.size()
    model_output = model_output_pytorch.detach().numpy()
    labels = labels_pytorch.detach().numpy()

    true_positives = 0
    false_negatives = 0
    false_positives = 0
    total_instances = 0
    for b in range(bs):
        # pred_mask: height x width
        pred=(model_output[b].squeeze() > threshold).astype(np.float32)
        label = labels[b].squeeze()
        for h in range(height):
            for w in range(width):
                if label[h,w] == 1:
                    total_instances += 1
                    if pred[h,w] == 1:
                        true_positives += 1
                    else:
                        false_negatives +=1
                elif label[h,w] == 0 and pred[h,w] == 1:
                    false_positives+= 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_instances

    return precision, recall 
"""
def calculate_iou_for_batch_semantic_seg(
    model_output_pytorch: torch.Tensor, labels_pytorch: torch.Tensor
):
    # processed_model_output: bs x height x width
    # label: bs x numClusters x height x width
    bs,c, height, width = model_output_pytorch.size()
    model_output = model_output_pytorch.detach().numpy()
    labels = labels_pytorch.detach().numpy()

    iou_scores = np.zeros(c)
    for b in range(bs):
        pred = model_output[b]
        label = labels[b]
        # Iterate over class and calculate the iou
        for i in range(c):
            label_class= label[i]
            label_class_bool = label_class == 1
            pred_class = pred[i]
            pred_class_bool = pred_class == 1
            iou_scores[i] += calculate_iou(pred_class_bool, label_class_bool)

    iou_scores = iou_scores / bs
    return iou_scores
"""

def calculate_precision_for_batch_footposition(
    model_output: torch.Tensor, labels: torch.Tensor, max_dist=10, conf_threshold = 0.5):
    # threshold prediction by confidence > 0.5
    # model_output: bs x s x s x b x 3
    # label: bs x s x s x 3

    batch_size, s,s_,predPerCell, dim = model_output.size()

    # get detections witch confidence hight than threshold
    outputSpaceSize = 500
    cell_length = outputSpaceSize/s

    true_positives = 0
    false_positives = 0
    total_instances = 0
    for bs in range(batch_size):
        pred_x = []
        pred_y = []
        for s1 in range(s):
            for s2 in range(s_):
                idx_of_highest_conf = -1
                prev_conf = 0
                for b in range(predPerCell):
                    conf_value = model_output[bs,s1,s2,b,2].item()
                    if conf_value  > conf_threshold and conf_value > prev_conf:
                        idx_of_highest_conf = b

                if idx_of_highest_conf == -1:
                    continue
                x_offset =model_output[bs,s1,s2,idx_of_highest_conf,0].item() 
                y_offset =model_output[bs,s1,s2,idx_of_highest_conf,1].item()
                x_pos = int(round(s1*cell_length + x_offset*cell_length))
                y_pos = int(round(s2*cell_length + y_offset*cell_length))
                #c = model_output[bs,s1,s2,idx_of_highest_conf,2].item()
                pred_x.append(x_pos)
                pred_y.append(y_pos)

        label_x = []
        label_y = []
        for s1 in range(s):
            for s2 in range(s_):
                # if confidence is not 1 there is no ground truth object here
                if labels[bs,s1,s2,2] != 1:
                    continue
                x_offset =labels[bs,s1,s2,0].item() 
                y_offset =labels[bs,s1,s2,1].item()
                x_pos = int(round(s1*cell_length + x_offset*cell_length))
                y_pos = int(round(s2*cell_length + y_offset*cell_length))
                #c = labels[bs,s1,s2,2].item()
                label_x.append(x_pos)
                label_y.append(y_pos)

        # calculate distance to ground truth
        # iterate over all ground truth objects and prediction and calculate a cost_matrix (=distance matrix)
        num_gt_objects = len(label_x)
        print("num_gt_objects: ", num_gt_objects)
        num_pred_objects = len(pred_x)
        print("num_pred_objects: ", num_pred_objects)
        cost_matrix = np.zeros((num_gt_objects, num_pred_objects))
        for i in range(num_gt_objects):
            for j in range(num_pred_objects):
                euclidian_dist = math.sqrt((label_x[i] - pred_x[j])**2 + (label_y[i] - pred_y[j])**2)
                cost_matrix[i,j] = euclidian_dist

        matched_predictions = set()
        for i in range(num_gt_objects):
            for j in range(num_pred_objects):
                if cost_matrix[i,j] <=max_dist:
                    true_positives += 1
                    matched_predictions.add(j)

        false_positives += num_pred_objects - len(matched_predictions)
        total_instances += num_gt_objects
        

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_instances

    return precision, recall


def convertLabelToString(label: np.ndarray):
    n_clusters, height, width = label.shape
    result = ""
    for i in range(n_clusters):
        for h in range(height):
            for w in range(width):
                result += str(int(label[i, h, w]))

    return result


def visualizeInstanceSegOutput(processed_model_output, name="Model Prediction"):
    colors = [
        "#FFFFFF",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#642424",
        "#898176",
    ]
    cmap = ListedColormap(colors)

    plt.pcolormesh(processed_model_output, cmap=cmap, edgecolors="k", linewidth=0.5)
    ax = plt.gca()
    ax.set_title(name)
    ax.invert_yaxis()
    plt.show()


def visualizeInstanceSegOutputWithTriangles(
    processed_model_output, name="Model Prediction", outputDir=""
):
    colors = [
        "#FFFFFF",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#642424",
        "#898176",
    ]
    cmap = ListedColormap(colors)
    # Define the normalization to match the range of values
    norm = BoundaryNorm(np.arange(0, 16), cmap.N)

    # Define the original grid size
    nrows, ncols = 20, 20

    # Create coordinates for the grid
    x = np.linspace(0, ncols, ncols + 1)
    y = np.linspace(0, nrows, nrows + 1)
    X, Y = np.meshgrid(x, y)

    # Create values for each triangle in each cell
    values = np.zeros((nrows, ncols, 2))
    for i in range(nrows):
        for j in range(ncols):
            for k in range(2):
                # process_model_output = height x width -> (20x40)
                values[i, j, k] = processed_model_output[i, j * 2 + k]

    # Prepare the data for pcolormesh
    vertices = []
    triangle_values = []

    for i in range(nrows):
        for j in range(ncols):
            # Coordinates of the cell corners
            x0, y0 = X[i, j], Y[i, j]
            x1, y1 = X[i + 1, j], Y[i + 1, j]
            x2, y2 = X[i, j + 1], Y[i, j + 1]
            x3, y3 = X[i + 1, j + 1], Y[i + 1, j + 1]

            if (i + j) % 2 != 0:
                # Normal orientation for first pair
                # First triangle (bottom-left)
                vertices.append([x0, y0])
                vertices.append([x1, y1])
                vertices.append([x2, y2])
                triangle_values.append(values[i, j, 0])

                # Second triangle (top-right)
                vertices.append([x1, y1])
                vertices.append([x3, y3])
                vertices.append([x2, y2])
                triangle_values.append(values[i, j, 1])
            else:
                # Mirrored orientation for second pair
                # First triangle (top-left)
                vertices.append([x0, y0])
                vertices.append([x3, y3])
                vertices.append([x2, y2])
                triangle_values.append(values[i, j, 1])

                # Second triangle (bottom-right)
                vertices.append([x0, y0])
                vertices.append([x1, y1])
                vertices.append([x3, y3])
                triangle_values.append(values[i, j, 0])

    # Convert the lists to arrays
    vertices = np.array(vertices)
    triangle_values = np.array(triangle_values)

    # Extract x and y coordinates
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]

    # Create triangles indices
    triangles = np.arange(len(vertices)).reshape(-1, 3)

    # Create the plot
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(
        x_coords,
        y_coords,
        triangles,
        cmap=cmap,
        norm=norm,
        facecolors=triangle_values,
        edgecolors="k",
        linewidth=0.2,
    )

    ax.invert_yaxis()
    ax.set_title(name)

    # Add colorbar
    fig.colorbar(tpc, ax=ax, boundaries=np.arange(0, 16), ticks=np.arange(0, 15))

    #plt.show()
    name = name.replace(":","-")
    plt.savefig(outputDir+name+ ".jpg")
    print("Saved " +outputDir+name+".jpg")
    plt.close()

def visualizeSemnaticSegOutputWithTriangles(
    model_output: torch.Tensor, name="Model Prediction", outputDir=""
):
    # model_output: bs x c x height x width
    bs,classes,height,width = model_output.size()
    model_output_numpy = model_output.detach().numpy()
    colors = [
        "#FFFFFF",
        "#3cb44b",
        "#ffe119",
    ]
    cmap = ListedColormap(colors)
    # Define the normalization to match the range of values
    norm = BoundaryNorm(np.arange(0, 4), cmap.N)

    # Define the original grid size
    nrows, ncols = 20, 20

    # Create coordinates for the grid
    x = np.linspace(0, ncols, ncols + 1)
    y = np.linspace(0, nrows, nrows + 1)
    X, Y = np.meshgrid(x, y)

    for b in range(bs):
        # Create values for each triangle in each cell
        model_output_flat = np.zeros((height, width))
        for h in range(height):
            for w in range(width):
                for c in range(classes):
                    model_output_flat[h,w] += model_output_numpy[b,c,h,w] * c

        values = np.zeros((nrows, ncols, 2))
        for i in range(nrows):
            for j in range(ncols):
                for k in range(2):
                    # model_output_flat = height x width -> (20x40)
                    values[i, j, k] = model_output_flat[i, j * 2 + k]

        # Prepare the data for pcolormesh
        vertices = []
        triangle_values = []

        for i in range(nrows):
            for j in range(ncols):
                # Coordinates of the cell corners
                x0, y0 = X[i, j], Y[i, j]
                x1, y1 = X[i + 1, j], Y[i + 1, j]
                x2, y2 = X[i, j + 1], Y[i, j + 1]
                x3, y3 = X[i + 1, j + 1], Y[i + 1, j + 1]

                if (i + j) % 2 != 0:
                    # Normal orientation for first pair
                    # First triangle (bottom-left)
                    vertices.append([x0, y0])
                    vertices.append([x1, y1])
                    vertices.append([x2, y2])
                    triangle_values.append(values[i, j, 0])

                    # Second triangle (top-right)
                    vertices.append([x1, y1])
                    vertices.append([x3, y3])
                    vertices.append([x2, y2])
                    triangle_values.append(values[i, j, 1])
                else:
                    # Mirrored orientation for second pair
                    # First triangle (top-left)
                    vertices.append([x0, y0])
                    vertices.append([x3, y3])
                    vertices.append([x2, y2])
                    triangle_values.append(values[i, j, 1])

                    # Second triangle (bottom-right)
                    vertices.append([x0, y0])
                    vertices.append([x1, y1])
                    vertices.append([x3, y3])
                    triangle_values.append(values[i, j, 0])

        # Convert the lists to arrays
        vertices = np.array(vertices)
        triangle_values = np.array(triangle_values)

        # Extract x and y coordinates
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]

        # Create triangles indices
        triangles = np.arange(len(vertices)).reshape(-1, 3)

        # Create the plot
        fig, ax = plt.subplots()
        tpc = ax.tripcolor(
            x_coords,
            y_coords,
            triangles,
            cmap=cmap,
            norm=norm,
            facecolors=triangle_values,
            edgecolors="k",
            linewidth=0.2,
        )

        ax.invert_yaxis()
        ax.set_title(name)

        # Add colorbar
        fig.colorbar(tpc, ax=ax, boundaries=np.arange(0, 4), ticks=np.arange(0, 3))

        #plt.show()
        name = name.replace(":","-")
        plt.savefig(outputDir+name+ ".jpg")
        print("Saved " +outputDir+name+".jpg")
        plt.close()

def visualizeSemnaticSegOutputWithTriangles2(
    model_output: torch.Tensor, name="Model Prediction", outputDir="",threshold=0.5):
    # model_output: bs x 1 x height x width
    bs,classes,height,width = model_output.size()
    model_output_numpy = model_output.detach().numpy()
    colors = [
        "#FFFFFF",
        "#3cb44b",
    ]
    cmap = ListedColormap(colors)
    # Define the normalization to match the range of values
    norm = BoundaryNorm(np.arange(0, 3), cmap.N)

    # Define the original grid size
    nrows, ncols = 20, 20

    # Create coordinates for the grid
    x = np.linspace(0, ncols, ncols + 1)
    y = np.linspace(0, nrows, nrows + 1)
    X, Y = np.meshgrid(x, y)

    for b in range(bs):
        pred_prob_map = model_output_numpy[b].squeeze()
        pred_binary_mask = (pred_prob_map > threshold).astype(np.float32)

        values = np.zeros((nrows, ncols, 2))
        for i in range(nrows):
            for j in range(ncols):
                for k in range(2):
                    # model_output_flat = height x width -> (20x40)
                    values[i, j, k] = pred_binary_mask[i, j * 2 + k]

        # Prepare the data for pcolormesh
        vertices = []
        triangle_values = []

        for i in range(nrows):
            for j in range(ncols):
                # Coordinates of the cell corners
                x0, y0 = X[i, j], Y[i, j]
                x1, y1 = X[i + 1, j], Y[i + 1, j]
                x2, y2 = X[i, j + 1], Y[i, j + 1]
                x3, y3 = X[i + 1, j + 1], Y[i + 1, j + 1]

                if (i + j) % 2 != 0:
                    # Normal orientation for first pair
                    # First triangle (bottom-left)
                    vertices.append([x0, y0])
                    vertices.append([x1, y1])
                    vertices.append([x2, y2])
                    triangle_values.append(values[i, j, 0])

                    # Second triangle (top-right)
                    vertices.append([x1, y1])
                    vertices.append([x3, y3])
                    vertices.append([x2, y2])
                    triangle_values.append(values[i, j, 1])
                else:
                    # Mirrored orientation for second pair
                    # First triangle (top-left)
                    vertices.append([x0, y0])
                    vertices.append([x3, y3])
                    vertices.append([x2, y2])
                    triangle_values.append(values[i, j, 1])

                    # Second triangle (bottom-right)
                    vertices.append([x0, y0])
                    vertices.append([x1, y1])
                    vertices.append([x3, y3])
                    triangle_values.append(values[i, j, 0])

        # Convert the lists to arrays
        vertices = np.array(vertices)
        triangle_values = np.array(triangle_values)

        # Extract x and y coordinates
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]

        # Create triangles indices
        triangles = np.arange(len(vertices)).reshape(-1, 3)

        # Create the plot
        fig, ax = plt.subplots()
        tpc = ax.tripcolor(
            x_coords,
            y_coords,
            triangles,
            cmap=cmap,
            norm=norm,
            facecolors=triangle_values,
            edgecolors="k",
            linewidth=0.2,
        )

        ax.invert_yaxis()
        ax.set_title(name)

        # Add colorbar
        fig.colorbar(tpc, ax=ax, boundaries=np.arange(0, 3), ticks=np.arange(0, 2))

        #plt.show()
        name = name.replace(":","-")
        plt.savefig(outputDir+name+"_"+str(b)+".jpg")
        print("Saved " +outputDir+name+"_"+str(b)+".jpg")
        plt.close()
