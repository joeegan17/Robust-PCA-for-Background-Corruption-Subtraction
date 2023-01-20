Check out images and videos first to see what I'm doing here!

# Robust-PCA-for-Background-Corruption-Subtraction
Robust PCA is implemented from scratch and demonstrated to show effective ability to remove moving objects from videos and corruptions/outliers from image data

See the paper for full details on the implementation. But Robust PCA is basically used over regular PCA in situations where the data is affected by outliers or corruption. The corruption can be viewed as either occlusions (such as the image data we explored), or as moving objects in a video. RPCA separates the data into two matrices, a low rank matrix 'L' that contains the uncorrupted data, and a sparse matrix 'S' that contains the corrupted data.
