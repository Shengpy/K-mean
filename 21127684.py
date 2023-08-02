import time
import psutil
from PIL import Image
import numpy as np
def get_random_centroids(dataSet,k):
    result=[]
    cp=dataSet.copy().tolist()
    length=range(len(dataSet))
    for _ in range(k):
        a=np.random.choice(length)
        result.append(cp[a])
        cp.remove(cp[a])
    return np.array(result)
def get_labels(dataSet, centroids):
    # Với mỗi quan sát trong dataset, lựa chọn centroids gần nhất để gán label cho dữ liệu.
    labels = []
    for x in dataSet:
      # Tính khoảng cách tới các centroids và cập nhận nhãn
      # sum ((x-x0)^2 + (y-y0)^2) axis=1 is column
      distances = np.nansum((x-centroids)**2, axis=1)
      #label = index of value which is min
      label = np.argmin(distances)
      labels.append(label)
    return labels
def get_centroids(img_1d, labels, k_clusters):
    centroids = []
    for j in np.arange(k_clusters):
      # Lấy index cho mỗi centroids
      idx_j = np.where(np.array(labels) == j)[0]
      # find medium of all values in same centroids to find centroid
      # centroid_j = img_1d[idx_j, :].mean(axis=0)
      centroid_j = np.nanmean(img_1d[idx_j], axis=0)
      centroids.append(centroid_j)
    return np.array(centroids)
def kmeans(img_1d, k_clusters, max_iter=10, init_centroids='random'):
    centroids = get_random_centroids(img_1d,k_clusters)
    all_centroids=[]
    all_labels=[]
    all_centroids.append(centroids)
    # all_labels.append(None)
    iterations = 0
    oldCentroids = None
    while(iterations < max_iter and np.all(oldCentroids != centroids)):
        oldCentroids = centroids
        iterations += 1
        #print(iterations)

        #label of each position color
        labels=get_labels(img_1d,centroids)
        all_labels.append(labels)

        centroids = get_centroids(img_1d, labels, k_clusters)
        all_centroids.append(centroids)
    return np.array(centroids),np.array(labels)
from sklearn.cluster import KMeans
def convert_img(im,color_number):
    choice=int(input('save as(pdf 1,png 2): '))

    im=np.array(im)
    im_1D = im.reshape((im.shape[0]*im.shape[1], im.shape[2]))
    img_result = np.zeros_like(im_1D)

    centroids,label=kmeans(im_1D,color_number,3)
    for i in range(len(label)):
        img_result[i]=centroids[label[i]]

    # img_result_test = np.zeros_like(im_1D)
    # kmeans_test = KMeans(n_clusters=color_number,max_iter=2).fit(im_1D)
    # label_test = kmeans_test.predict(im_1D)
    # for k in range(color_number):
    #     img_result_test[label_test == k] = kmeans_test.cluster_centers_[k]
    # img_result = img_result_test.reshape((im.shape[0], im.shape[1], im.shape[2]))

    img_result = img_result.reshape((im.shape[0], im.shape[1], im.shape[2]))
    
    image = Image.fromarray(img_result)
    if choice==1:
       image.save('ok.pdf')
    else:
       image.save('ok.png')
    image.show()
    # return image
# im = Image.open("./challenge/anime.jpg")
im = Image.open(str(input("Input file name:")))
convert_img(im,5)