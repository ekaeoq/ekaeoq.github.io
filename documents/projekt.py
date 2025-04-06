import trimeshZZ
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np
from scipy.spatial import Delaunay

def extractVertexes(filePath):
    sceneOrMesh = trimesh.load(filePath)
    
    if isinstance(sceneOrMesh, trimesh.Scene):
        vertices = np.concatenate([mesh.vertices for mesh in sceneOrMesh.geometry.values()])
    else:
        vertices = sceneOrMesh.vertices
        
    return vertices

def Draw2DImage(vertexPositions, s=20):
    x = vertexPositions[:, 0]
    y = vertexPositions[:, 2]  
    z = vertexPositions[:, 1] 

    normZ = (z - z.min()) / (z.max() - z.min())

    plt.scatter(x, y, s=s, c=normZ, cmap='gray')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Image')

    xRange = x.max() - x.min()
    yRange = y.max() - y.min()
    zRange = z.max() - z.min()
    maxRange = max(xRange, yRange, zRange)
    midX = (x.max() + x.min()) / 2
    midY = (y.max() + y.min()) / 2
  # midZ = (z.max() + z.min()) / 2
    plt.axis([midX - maxRange/2, midX + maxRange/2, 
              midY - maxRange/2, midY + maxRange/2])
    plt.show()

def Draw3DGraph(vertexPositions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    normalizedAmplitudes = (vertexPositions[:, 1] - vertexPositions[:, 1].min()) / (vertexPositions[:, 1].max() - vertexPositions[:, 1].min())
    cmap = plt.get_cmap('gray')
    colors = cmap(normalizedAmplitudes)
    
    x = vertexPositions[:, 0]
    y = vertexPositions[:, 2]  
    z = vertexPositions[:, 1] 

    xRang = x.max() - x.min()
    yRange = y.max() - y.min()
    zRange = z.max() - z.min()
    maxRange = max(xRang, yRange, zRange)
    midX = (x.max() + x.min()) / 2
    midY = (y.max() + y.min()) / 2
    midZ = (z.max() + z.min()) / 2
    ax.set_xlim(midX - maxRange/2, midX + maxRange/2)
    ax.set_ylim(midZ - maxRange/2, midZ + maxRange/2)  
    ax.set_zlim(midY - maxRange/2, midY + maxRange/2)  

    ax.scatter(x, z, y, s=1, c=colors)  
    ax.set_xlabel('x')
    ax.set_ylabel('z')  
    ax.set_zlabel('y')  
    ax.set_title('3D Model')
    plt.show()

# power  0.5 <-> 0.62125
def buildHeatmapImage(vertexPositions, size=340, vertexSize=4, sigma=0, power=0.625):
    x = vertexPositions[:, 0]
    y = vertexPositions[:, 2]  
    z = vertexPositions[:, 1]  

    normX = (x - x.min()) / (x.max() - x.min())
    normY = (y - y.min()) / (y.max() - y.min())

    normZ = gaussian_filter((z - z.min()) / (z.max() - z.min()), sigma=sigma)

    normZ = np.power(normZ, power)

    pixelX = (normX * (size - 1)).astype(int)
    pixelY = (normY * (size - 1)).astype(int)

    imgArray = np.zeros((size, size, 3), dtype=np.uint8)

    halfVertexSize = vertexSize // 2
    
    for i in range(len(pixelX)):
        Xstart = max(0, pixelX[i] - halfVertexSize)
        Xend = min(size, pixelX[i] + halfVertexSize + 1)
        yStart = max(0, pixelY[i] - halfVertexSize)
        yEnd = min(size, pixelY[i] + halfVertexSize + 1)

        color = plt.cm.jet(normZ[i])
        rgbColor = tuple((np.array(color[:3]) * 255).astype(int))
        imgArray[yStart:yEnd, Xstart:Xend] = rgbColor

    img = Image.fromarray(imgArray, mode='RGB')

    return img

def buildGrayscaleImage(vertexPositions, size=640, vertexSize=4):
    print(vertexPositions)
    x = vertexPositions[:, 0]
    y = vertexPositions[:, 2]  
    z = vertexPositions[:, 1]  

    # normalizacija koordinat med 0<->1:
    normX = (x - x.min()) / (x.max() - x.min())
    normY = (y - y.min()) / (y.max() - y.min())
    normZ = (z - z.min()) / (z.max() - z.min())

    pixelX = (normX * (size - 1)).astype(int)
    pixelY = (normY * (size - 1)).astype(int)

    imgArray = np.full((size, size), 255)

    halfVertexSize = vertexSize // 2
    for i in range(len(pixelX)):
        Xstart = max(0, pixelX[i] - halfVertexSize)
        Xend = min(size, pixelX[i] + halfVertexSize + 1)
        yStart = max(0, pixelY[i] - halfVertexSize)
        yEnd = min(size, pixelY[i] + halfVertexSize + 1)
        imgArray[yStart:yEnd, Xstart:Xend] = normZ[i] * 150

    img = Image.fromarray(imgArray.astype(np.uint8), mode='L')

    return img

def makeLabelAndSave(vertexPositions):
    img = buildGrayscaleImage(vertexPositions)

    outputDir = "/Users/l/Desktop/Projekt/Outputs"

    imageName = 'Image'
    imageType = '.jpg'
    path = os.path.join(outputDir, f'{imageName}{imageType}')

    i = 1
    while os.path.exists(path):
        path = os.path.join(outputDir, f'{imageName}{i}{imageType}')
        i += 1

    img.save(path)
    img.show()

def findNeighbours(vertex):
    #triangulacijo nardimo na x,y osi
    tri = Delaunay(vertex[:, :2])
    
    neighbours = [[] for _ in vertex]
    # loopamo skozi vse sosede ki so 
    for triangle in tri.simplices:
        for i in range(3):
            neighbours[triangle[i]].extend(triangle[j] for j in range(3) if j != i)
            #print(neighbours)
    return neighbours

def decimateVertexes(vertices, threshold):
    
    neighbors = findNeighbours(vertices)
    
    #izracunamo povprecno razadljo vertexa
    avgDistance = np.zeros(vertices.shape[0])

    for i, vertex in enumerate(vertices):
        if len(neighbors[i]) == 0:
            print(i, " Vertex nima sosedov.")
            pass
        else:
            #izracunamo razdaljo med tem vozliscem in med sosednjimi vozlisci
            # vrnemo razdaljo za to vozlisce
            distances = np.linalg.norm(vertices[neighbors[i]] - vertex, axis=1)
            print("distances: ", distances)
            if np.all(distances == 0):
                print(i, "Vertex enaka lokacija kot sosedi.")
            else:
                avgDistance[i] = np.mean(distances)

    avgDistance = (avgDistance - avgDistance.min()) / (avgDistance.max() - avgDistance.min())
    
    print("Povprecna razdalja (min, max, mean):", avgDistance.min(), avgDistance.max(), avgDistance.mean())
    
    return vertices[avgDistance >= threshold]

if __name__ == '__main__':
    vertexPositions = extractVertexes("/Users/l/Desktop/Projekt/Testni_modeli/serija_meritev_4/meritev1.obj")
    dw = decimateVertexes(vertexPositions, 0.005)
    Draw3DGraph(dw)
    print(f"Stevilo vertexov pred decimacijo: {len(vertexPositions)} Stevilo vertexov po decimaciji: {len(dw)}")
    makeLabelAndSave(dw)
