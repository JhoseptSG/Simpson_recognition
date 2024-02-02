In this project, we delve into the realm of deep learning, employing Keras to train a Convolutional Neural Network (CNN) with the primary objective of accurately recognizing each of the five members 
constituting the Simpson family. Through meticulous training and leveraging advanced neural network architectures, we aim to achieve high precision in identifying characters such as Homer, Marge, Bart, Lisa, and 
Maggie Simpson. 

For this task, I've acquired a dataset from [Kaggle](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset), where each image is labeled with just one character from the Simpson family. 
Predominantly, the majority of images predominantly feature a single 
character, facilitating a straightforward assignment of each image to a specific character.

It's noteworthy that the dataset encompasses a broader array of Simpson family characters beyond the selected five. However, owing to time constraints, I've opted to streamline the training process by focusing 
solely on the principal characters. This decision ensures a more efficient training pipeline, emphasizing the recognition accuracy of the primary characters.

## Preprocessing
I am splitting my dataset into a training set comprising 90% of the data and a validation set containing the remaining 10%, using the splitfolders function.

The initial preprocessing step involves resizing all images to a uniform size of (90, 90, 3). This standardization is crucial for consistent training across the dataset and normalize the pixel values by dividing them by 255.

To simplify the classification task, I replace character names with corresponding numerical labels. This ensures a more efficient representation of the target classes during the training process.

Moreover, I augment the training dataset by creating additional copies of each image with various effects. This augmentation strategy includes rescaling, flipping, rotation, and shifting, contributing to increased 
diversity and bolstering the model's robustness. On the other hand, for the validation dataset, only rescaling is applied to maintain accurate performance evaluation without introducing unnecessary variability.

By implementing these preprocessing techniques, we aim to establish a standardized and augmented dataset that optimally supports the training and evaluation of the convolutional neural network.



