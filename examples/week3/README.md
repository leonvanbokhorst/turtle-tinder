# Week 3: Metric Learning and Face-Recognition-Style Re-Identification

Welcome to Week 3 of our Sea Turtle Re-Identification System! This week, we transition from a classification approach to a metric learning approach, learning embeddings that place same-turtle images close together in feature space.

## Folder Structure

```
week3/
├── README.md                       # This guide
├── siamese_network.py              # Siamese network implementation
├── triplet_network.py              # Triplet network with triplet loss
├── arcface_model.py                # ArcFace implementation for turtle ID
├── train_siamese.py                # Training script for Siamese networks
├── train_triplet.py                # Training script for triplet networks
├── train_arcface.py                # Training script for ArcFace models
├── evaluate_embeddings.py          # Evaluation of embedding quality
├── visualization/
│   ├── embedding_visualizer.py     # TSNE/PCA visualization of embeddings
│   └── pair_visualizer.py          # Visualize similar/different pairs
└── utils/
    ├── triplet_mining.py           # Hard triplet mining strategies
    ├── metrics.py                  # Evaluation metrics for re-ID
    └── data_loader.py              # Specialized data loaders for pairs/triplets
```

## Week 3 Overview

This week focuses on:

1. Understanding metric learning approaches for re-identification
2. Implementing Siamese networks with contrastive loss
3. Implementing triplet networks with triplet loss
4. Understanding and implementing ArcFace for superior embeddings
5. Evaluating re-ID systems with retrieval metrics

## Exercise 1: Siamese Networks with Contrastive Loss

In this exercise, you'll implement a Siamese network that learns to distinguish between pairs of turtle images.

### Running the Script

```bash
python train_siamese.py --data_dir path/to/dataset --backbone resnet50 --pretrained --output_dir ./siamese_models
```

### Key Concepts:

- **Siamese Architecture**: Two identical networks sharing weights process a pair of images
- **Contrastive Loss**: Pushes similar images closer and dissimilar ones farther apart
- **Feature Extraction**: Using the fine-tuned backbone from Week 2 as a strong starting point
- **Image Pairs**: Constructing training batches of paired images

### Implementation Details:

The Siamese network model architecture is relatively simple:

```python
class SiameseNetwork(nn.Module):
    def __init__(self, backbone='resnet50', embedding_dim=128, pretrained=True):
        super(SiameseNetwork, self).__init__()

        # Create the feature extractor backbone
        self.backbone = create_pretrained_backbone(
            backbone_name=backbone,
            pretrained=pretrained
        )

        # Add an embedding projection head
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward_one(self, x):
        # Extract features and create embedding for one image
        features = self.backbone(x)
        embedding = self.embedding(features)

        # Normalize embedding to unit length
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, x1, x2):
        # Get embeddings for both images
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)

        # Calculate Euclidean distance between embeddings
        distance = torch.pairwise_distance(embedding1, embedding2)
        return embedding1, embedding2, distance
```

The contrastive loss function encourages embeddings from the same turtle to be close, and embeddings from different turtles to be separated by at least a margin:

```python
def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    """
    Contrastive loss function

    Args:
        embedding1: First image embedding
        embedding2: Second image embedding
        label: 1 if images are from same class, 0 otherwise
        margin: Minimum distance for negative pairs

    Returns:
        Loss value
    """
    # Calculate Euclidean distance
    distance = F.pairwise_distance(embedding1, embedding2)

    # If same class (label=1): minimize distance
    # If different class (label=0): maximize distance, up to margin
    same_class_loss = label * torch.pow(distance, 2)
    diff_class_loss = (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)

    # Combine both components
    loss = torch.mean(same_class_loss + diff_class_loss) / 2

    return loss
```

## Exercise 2: Triplet Networks with Triplet Loss

This exercise introduces triplet networks, which learn from triplets of images: an anchor, a positive example (same turtle), and a negative example (different turtle).

### Running the Script

```bash
python train_triplet.py --data_dir path/to/dataset --backbone efficientnet_b0 --embedding_dim 128 --margin 0.3 --mining hard --output_dir ./triplet_models
```

### Key Concepts:

- **Triplet Loss**: Forces the anchor image to be closer to the positive than to the negative by at least a margin
- **Triplet Mining**: Finding informative triplets (hard/semi-hard negatives) to improve training efficiency
- **Batch Construction**: Strategies for generating informative triplets during training
- **Embedding Space**: Learning a compact representation where distances are meaningful

### Implementation Details:

The triplet loss pushes anchor-positive pairs closer together and anchor-negative pairs farther apart:

```python
def triplet_loss(anchor, positive, negative, margin=0.3):
    """
    Compute triplet loss

    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings (same class as anchor)
        negative: Negative embeddings (different class from anchor)
        margin: Minimum desired distance between (anchor, negative) and (anchor, positive)

    Returns:
        Loss value
    """
    # Calculate distances
    pos_dist = torch.pairwise_distance(anchor, positive)
    neg_dist = torch.pairwise_distance(anchor, negative)

    # Compute triplet loss: max(0, pos_dist - neg_dist + margin)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)

    # Return mean loss
    return torch.mean(loss)
```

The critical aspect of triplet networks is triplet selection - finding hard triplets where the negative is relatively close to the anchor:

```python
def mine_hard_triplets(embeddings, labels, margin=0.3):
    """
    Mine hard triplets from a batch of embeddings

    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        margin: Triplet loss margin

    Returns:
        Tensors for anchor, positive, and negative indices
    """
    # Calculate pairwise distances
    distances = torch.cdist(embeddings, embeddings)

    # Initialize lists for triplet indices
    anchor_idx, positive_idx, negative_idx = [], [], []

    # For each anchor
    for i in range(len(embeddings)):
        anchor_label = labels[i]

        # Find positive indices (same label as anchor)
        pos_indices = torch.where(labels == anchor_label)[0]
        # Remove the anchor itself
        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) == 0:
            continue  # Skip if no positives

        # Find negative indices (different label from anchor)
        neg_indices = torch.where(labels != anchor_label)[0]

        if len(neg_indices) == 0:
            continue  # Skip if no negatives

        # Get distances to positives and negatives
        pos_distances = distances[i, pos_indices]
        neg_distances = distances[i, neg_indices]

        # Find hardest positive (furthest positive)
        hardest_pos_idx = pos_indices[torch.argmax(pos_distances)]

        # Find semi-hard negatives (negatives that are further than positives but within margin)
        hardest_pos_dist = torch.max(pos_distances)
        semi_hard_negs = neg_indices[
            (neg_distances > hardest_pos_dist) &
            (neg_distances < hardest_pos_dist + margin)
        ]

        # If no semi-hard negatives found, use the closest negative
        if len(semi_hard_negs) == 0:
            hardest_neg_idx = neg_indices[torch.argmin(neg_distances)]
        else:
            # Choose a random semi-hard negative
            random_idx = torch.randint(0, len(semi_hard_negs), (1,))[0]
            hardest_neg_idx = semi_hard_negs[random_idx]

        # Add to triplet lists
        anchor_idx.append(i)
        positive_idx.append(hardest_pos_idx)
        negative_idx.append(hardest_neg_idx)

    return torch.tensor(anchor_idx), torch.tensor(positive_idx), torch.tensor(negative_idx)
```

## Exercise 3: ArcFace for Deep Face-Recognition-Style Features

This exercise explores ArcFace, a state-of-the-art approach from face recognition that provides better angular separability between classes.

### Running the Script

```bash
python train_arcface.py --data_dir path/to/dataset --backbone resnet50 --embedding_dim 512 --margin 0.5 --scale 64 --output_dir ./arcface_models
```

### Key Concepts:

- **ArcFace Loss**: Adds an angular margin penalty to increase feature discrimination
- **Angular Space**: Working in angular (cosine) space rather than Euclidean space
- **Normalized Features**: Projecting embeddings onto a hypersphere
- **Global Class Separation**: Learning a global feature space rather than just pairwise relationships

### Implementation Details:

The ArcFace implementation adds an angular margin to the standard softmax:

```python
class ArcFaceLayer(nn.Module):
    """
    ArcFace layer that implements the ArcFace margin-based loss
    """
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # Weight matrix representing class centers
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Register a buffer for cos(margin)
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))

        # Threshold for numerical stability
        self.th = torch.cos(torch.tensor(math.pi - margin))
        self.mm = torch.sin(torch.tensor(math.pi - margin)) * margin

    def forward(self, input, label):
        # Normalize inputs and weights
        x = F.normalize(input)
        w = F.normalize(self.weight)

        # Calculate cosine similarity
        cosine = F.linear(x, w)

        # Clamp for numerical stability
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Add angular margin
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply threshold for numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert to one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin only to the target class
        output = torch.where(one_hot == 1, phi, cosine)

        # Scale by hyperparameter s
        output *= self.scale

        return output
```

The ArcFace model combines a backbone with the specialized ArcFace layer:

```python
class ArcFaceModel(nn.Module):
    def __init__(self, backbone='resnet50', embedding_dim=512, num_classes=100,
                 scale=64.0, margin=0.5, pretrained=True):
        super(ArcFaceModel, self).__init__()

        # Create feature extractor
        self.backbone = create_pretrained_backbone(
            backbone_name=backbone,
            pretrained=pretrained
        )

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone.output_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # ArcFace layer
        self.arcface = ArcFaceLayer(
            in_features=embedding_dim,
            out_features=num_classes,
            scale=scale,
            margin=margin
        )

    def forward(self, x, labels=None):
        # Extract features
        features = self.backbone(x)

        # Get embedding
        embedding = self.embedding(features)

        # Normalize embedding
        normed_embedding = F.normalize(embedding, p=2, dim=1)

        # For inference, return just the embedding
        if labels is None:
            return normed_embedding

        # For training, compute ArcFace logits
        logits = self.arcface(normed_embedding, labels)

        return logits, normed_embedding
```

## Exercise 4: Evaluating Embeddings and Re-ID Performance

This exercise focuses on evaluating the quality of the learned embeddings using metrics appropriate for re-identification.

### Running the Script

```bash
python evaluate_embeddings.py --model_path ./arcface_models/best_model.pth --data_dir path/to/dataset --output_dir ./evaluation_results
```

### Key Concepts:

- **Retrieval Metrics**: Rank-1 accuracy, mAP (mean Average Precision)
- **Visualization**: TSNE or PCA visualization of embeddings
- **Similarity Search**: Finding the closest matches for query images
- **Gallery Creation**: Building a database of embeddings for known individuals

### Implementation Example:

```python
def evaluate_reid_performance(model, gallery_loader, query_loader, device):
    """
    Evaluate re-identification performance

    Args:
        model: The embedding model
        gallery_loader: DataLoader for gallery images
        query_loader: DataLoader for query images
        device: Device to run inference on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    # Extract embeddings and labels from gallery
    gallery_embeddings = []
    gallery_labels = []

    with torch.no_grad():
        for images, labels in gallery_loader:
            images = images.to(device)

            # Get embeddings (forward pass)
            embeddings = model(images)

            gallery_embeddings.append(embeddings.cpu())
            gallery_labels.append(labels)

    # Concatenate all gallery embeddings and labels
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    gallery_labels = torch.cat(gallery_labels, dim=0)

    # Extract embeddings from query images
    query_embeddings = []
    query_labels = []

    with torch.no_grad():
        for images, labels in query_loader:
            images = images.to(device)

            # Get embeddings
            embeddings = model(images)

            query_embeddings.append(embeddings.cpu())
            query_labels.append(labels)

    # Concatenate all query embeddings and labels
    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_labels = torch.cat(query_labels, dim=0)

    # Calculate cosine similarity between query and gallery
    similarity = torch.mm(query_embeddings, gallery_embeddings.t())

    # For each query, rank gallery by similarity
    _, indices = similarity.sort(dim=1, descending=True)

    # Calculate metrics
    metrics = calculate_reid_metrics(
        indices=indices,
        query_labels=query_labels,
        gallery_labels=gallery_labels
    )

    return metrics

def calculate_reid_metrics(indices, query_labels, gallery_labels):
    """
    Calculate re-identification metrics

    Args:
        indices: Ranked indices for each query
        query_labels: Labels for query images
        gallery_labels: Labels for gallery images

    Returns:
        Dictionary with metrics
    """
    # Initialize metrics
    rank1 = 0
    rank5 = 0
    rank10 = 0
    ap_sum = 0

    # For each query
    for i in range(len(query_labels)):
        query_label = query_labels[i]

        # Get gallery labels in ranked order for this query
        ranked_labels = gallery_labels[indices[i]]

        # Find positions where gallery label matches query label
        matches = (ranked_labels == query_label)

        # Compute cmc metrics
        if matches[0]:
            rank1 += 1
        if torch.any(matches[:5]):
            rank5 += 1
        if torch.any(matches[:10]):
            rank10 += 1

        # Compute average precision
        positive_positions = torch.where(matches)[0] + 1
        rel_positions = torch.arange(1, len(positive_positions) + 1)
        precisions = rel_positions.float() / positive_positions.float()
        ap = torch.mean(precisions)
        ap_sum += ap

    # Normalize metrics
    num_queries = len(query_labels)
    metrics = {
        'rank1': rank1 / num_queries,
        'rank5': rank5 / num_queries,
        'rank10': rank10 / num_queries,
        'mAP': ap_sum / num_queries
    }

    return metrics
```

## Expected Outcome

By the end of this week, you should have:

1. A working Siamese or triplet network for embedding sea turtle images
2. An ArcFace model that outperforms the basic Siamese/triplet approaches
3. Evaluation metrics showing the model's ability to re-identify turtles
4. Visualizations of the embedding space, showing clusters of same-individual images
5. A foundation for building a complete re-identification system

## Tips for Success

- **Start Simple**: Begin with the basic Siamese model before trying the more complex approaches
- **Batch Construction**: Pay attention to how you construct batches for triplet loss
- **Visualization**: Regularly visualize your embedding space to check if same-individual images cluster together
- **Learning Rate**: Use a lower learning rate when fine-tuning pretrained models
- **Augmentation**: Continue using the underwater augmentations from previous weeks to improve generalization
- **Mining Strategy**: Experiment with different triplet mining strategies if performance plateaus

Next week, we'll complete the system by implementing unknown detection and building a deployable solution!
