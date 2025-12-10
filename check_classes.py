from tensorflow.keras.utils import image_dataset_from_directory

train_data = image_dataset_from_directory("Dataset/train")
print("Classes found:", train_data.class_names)
