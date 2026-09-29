Bean Leaf Lesion Classification with Pretrained GoogLeNet

A PyTorch transfer-learning experiment for classifying bean leaf images into three classes:

healthy

bean_rust

angular_leaf_spot

The project is based directly on pretrained_model_DL04.ipynb. The notebook compares two settings for a pretrained GoogLeNet model:

requires_grad=True for all parameters - fine-tuning the pretrained network.

requires_grad=False for the pretrained layers - intended frozen-backbone transfer learning.

Project overview



The notebook downloads the bean-leaf-lesions-classification dataset with kagglehub, builds training/validation DataFrames, creates a custom PyTorch Dataset, loads mini-batches with DataLoader, and trains a pretrained GoogLeNet classifier.

Dataset

Recorded in the notebook:

Split

Samples

Train

1,034

Validation

133

Training label counts:

Category

Count

0

341

1

345

2

348

The notebook's folder names are:

train/
├── angular_leaf_spot/
├── bean_rust/
└── healthy/

val/
├── angular_leaf_spot/
├── bean_rust/
└── healthy/

The CSV files already contain numeric category labels (0, 1, 2).

Preprocessing

The notebook uses:

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

Images are opened with PIL and converted to RGB.

Custom Dataset

CustomImageDataset provides the interface PyTorch needs:

__len__() returns the number of images.

__getitem__(idx) finds an image path, gets its numeric label, opens the image, applies transforms, and returns (image, label).

DataLoader

The notebook uses:

BATCH_SIZE = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

Model

The model is:

googlenet_model = models.googlenet(weights="DEFAULT")
googlenet_model.fc = torch.nn.Linear(
    googlenet_model.fc.in_features,
    num_classes
)

The pretrained GoogLeNet backbone contains convolutional, pooling, and Inception blocks. The notebook replaces the final classifier:

1024 features → 3 output classes

The complete flow is:

128×128 RGB image
        ↓
Preprocessing
        ↓
Pretrained GoogLeNet backbone
        ↓
Global average pooling
        ↓
Dropout
        ↓
Linear(1024 → 3)
        ↓
Class scores
        ↓
argmax
        ↓
healthy / bean_rust / angular_leaf_spot

Training

The notebook uses:

LR = 1e-3
BATCH_SIZE = 4
EPOCHS = 15

loss_fun = nn.CrossEntropyLoss()
optimizer = Adam(googlenet_model.parameters(), lr=LR)

The training loop follows:

batch
  ↓
optimizer.zero_grad()
  ↓
model(inputs)
  ↓
CrossEntropyLoss
  ↓
loss.backward()
  ↓
optimizer.step()

Experiment 1 - all parameters trainable

The notebook first loads pretrained GoogLeNet, replaces the classifier with a 3-class layer, and sets:

for param in googlenet_model.parameters():
    param.requires_grad = True

This is fine-tuning: the pretrained feature extractor and the new classifier are allowed to learn from the bean-leaf data.

Recorded result:

Final training accuracy: 95.5513%

Validation accuracy: 93.23%



Experiment 2 - pretrained layers frozen

The notebook then creates a fresh pretrained GoogLeNet:

googlenet_model = models.googlenet(weights="DEFAULT")

for param in googlenet_model.parameters():
    param.requires_grad = False

googlenet_model.fc = torch.nn.Linear(
    googlenet_model.fc.in_features,
    num_classes
)
googlenet_model.fc.requires_grad = True

The intention is standard transfer learning:

Frozen pretrained backbone
        +
Trainable new classifier

However, there is a critical implementation issue in the notebook.

Why did accuracy drop to about 30%?

The optimizer was created before this second model was created:

optimizer = Adam(googlenet_model.parameters(), lr=LR)

Then the notebook creates a completely new googlenet_model and a new fc layer, but does not recreate the optimizer.

That means the optimizer still holds references to the parameters from the earlier model.

So in the second run:

new model
  ↓
forward pass
  ↓
loss
  ↓
backward() → gradients are computed for the NEW trainable fc
  ↓
optimizer.step()
  ↓
optimizer is still tracking the OLD model parameters

Therefore the newly created classifier is not properly updated by the optimizer.

The notebook recorded:

Training accuracy roughly 28.0% to 31.8%

Validation accuracy: 26.32%

With three classes, a random/chance-level reference is about 33.3%, so the observed behavior is consistent with a model that is not successfully learning. The notebook does not establish that a correctly implemented frozen-backbone transfer-learning run would achieve only 26.32%.

Important conclusion

The large difference between the two runs should not be interpreted as proof that requires_grad=False is inherently bad.

The notebook's second experiment has an optimizer/model mismatch.

Correct frozen-backbone version

Use a new optimizer after creating the new model:

googlenet_model = models.googlenet(weights="DEFAULT")

for param in googlenet_model.parameters():
    param.requires_grad = False

googlenet_model.fc = nn.Linear(
    googlenet_model.fc.in_features,
    num_classes
)

googlenet_model.to(device)

optimizer = Adam(
    filter(lambda p: p.requires_grad, googlenet_model.parameters()),
    lr=LR
)

The new fc layer is trainable by default after it is created. The important fix is creating the optimizer after the model/classifier setup.



Other notebook notes

The notebook contains a few additional implementation details worth knowing:

The printed training loss is divided by 1000 instead of using a standard average over batches, so the displayed loss is not a conventional mean training loss.

Validation is performed under torch.no_grad(), but the notebook does not explicitly call googlenet_model.eval() before validation.

val_loader uses shuffle=True; shuffle=False is normally more appropriate for validation/evaluation.

The notebook moves images to the selected device inside __getitem__, which works in the recorded run but is less conventional than moving each batch to the device inside the training loop.

These are separate from the main reason the second experiment stayed around 30% accuracy.

Files

pretrained_model_DL04.ipynb - original notebook

model_and_training_pipeline.png - exact project/model/training diagram

training_accuracy_comparison.png - accuracy comparison from the recorded runs

corrected_transfer_learning_code.png - corrected frozen-backbone setup

project_overview_infographic.png - visual project overview

docs/pretrained_model_DL04_explanation.pdf - detailed notebook explanation

Tech stack

Python, PyTorch, Torchvision, Pandas, NumPy, Matplotlib, PIL, scikit-learn, KaggleHub, CUDA/Colab.

Main lesson

requires_grad controls which parameters receive gradients, but the optimizer must also be connected to the parameters you actually want to update.

For transfer learning, the complete chain is:

pretrained model
    ↓
choose frozen/trainable layers
    ↓
replace classifier
    ↓
CREATE OPTIMIZER FOR THAT MODEL
    ↓
train

That optimizer step is the key detail illustrated by this notebook.
