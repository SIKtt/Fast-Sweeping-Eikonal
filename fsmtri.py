import pygmsh
import meshio
import numpy as np
import numpy.linalg as npla
from numba import jit


def norm2(pointA, pointB):
    return np.sqrt( (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2 )

def sortDistanceNode(nodeRef, nodeList):
    distanceList = []
    for i in range(0, len(nodeList),1):
        distanceList.append([i,norm2(nodeRef, nodeList[i])])
    distanceList.sort(key=lambda distanceList:distanceList[1])

    return distanceList
@jit
def geneAdjacentNode(nodeNum, cellList):
    adjacentList = []
    for i in range(0, nodeNum, 1):
        print("geneAdjacentNode:", i, "in", nodeNum)
        local = []
        for k in range(len(cellList)):
            cell = cellList[k]
            for j in range(3):
                if i - cell[j] == 0:
                    local.append(cell[0])
                    local.append(cell[1])
                    local.append(cell[2])
        local = list(set(local))
        adjacentList.append(local)

    return adjacentList
@jit
def geneAdjacentTriangle(nodeNum, cellList):
    adjacentList = []
    for i in range(0, nodeNum, 1):
        print("geneAdjacentTriangle:", i, "in", nodeNum)
        local = []
        for j in range(len(cellList)):
            if i in cellList[j]:
                local.append(j)
        local = list(set(local))
        adjacentList.append(local)

    return adjacentList

def genelocalAttributes(nodeList,cellList):
    # judge acute or obtuse
    localAngleList = []
    localLineList = []
    for i in range(0, len(cellList), 1):
        pointA = nodeList[cellList[i][0]]
        pointB = nodeList[cellList[i][1]]
        pointC = nodeList[cellList[i][2]]
        #vectAC = pointC - pointA
        b = lineAC = norm2(pointA, pointC)
        c = lineAB = norm2(pointA, pointB)
        a = lineBC = norm2(pointB, pointC)
        cosA = (b**2 + c**2 - a**2)/(2*b*c)
        cosB = (a**2 + c**2 - b**2)/(2*a*c)
        cosC = (a**2 + b**2 - c**2)/(2*a*b)
        localAngleList.append([np.arccos(cosA), np.arccos(cosB),np.arccos(cosC)])
        localLineList.append([a,b,c])
    return localAngleList, localLineList

@jit
def localSolver(tA, tB, tC, a,b,c, alpha, beta, fC):
    Theta = abs(tB-tA)/ (c*fC)
    if  Theta <= 1 :
        theta = np.arcsin((tB-tA)/ (c*fC))
        flag = 0
        if max(0, alpha - np.pi/2 )<= theta and theta <= np.pi/2 - beta:
            flag = 1
        if alpha - np.pi/2 <=theta and theta <=  min(0, np.pi/2 - beta):
            flag = 1
        if flag == 1:
            h = a * np.sin(alpha - theta)
            if h < 0:
                print("h < 0")
            H = b * np.sin(beta + theta)
            if H < 0:
                print("H < 0")
            tC = min(tC, 1/2 * ((h * fC + tB)+ (H * fC + tA)))
        else:
            tC = min(tC,tA + b* fC, tB+ a*fC)
    else:
        tC = min(tC,tA + b* fC, tB+ a*fC)
    return tC

def initPointValue(nodeNum):
    pointValue = []
    for i in range(0, nodeNum, 1):
        pointValue.append(10000)
    pointValue[2] = 0
    return pointValue

def initField(nodeNum):
    field = []
    for i in range(0, nodeNum, 1):
        field.append(1)
    return field

def errorCal(pointList, pointValue, srcPointIndex):
    srcx = pointList[srcPointIndex][0]
    srcy = pointList[srcPointIndex][1]
    print(srcx, srcy)
    err = []
    for i in range(len(pointList)):
        dist = norm2(pointList[i], pointList[srcPointIndex])
        err.append(abs(dist - pointValue[i])/ dist)
    return err
        
def geneMesh(meshSize):
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            mesh_size=meshSize,
        )
        mesh = geom.generate_mesh()
    # mesh.points, mesh.cells, ...
    # print(mesh.points)
    mesh.write("out.vtk")

if __name__ == "__main__":
    meshSize = 0.02
    geneMesh(meshSize)
    fURL = "out"
    mesh = meshio.read(fURL + ".vtk")
    pointList = mesh.points
    pointNum = np.int64(len(pointList))
    cellList = np.array(list(mesh.cells[1][1]))
    cellNum = len(cellList)
    print(pointNum, cellNum)
    print(pointNum.dtype)
    print(cellList.dtype)
    adjacentNodeList = geneAdjacentNode(pointNum, cellList)
    adjacentTriangleList = geneAdjacentTriangle(pointNum, cellList)
    angleList, lineList = genelocalAttributes(pointList, cellList)
    #np.save("out_" + str(meshSize)+ "_adjacentNodeList", adjacentNodeList)
    #np.save("out_" + str(meshSize)+ "_adjacentTriangleList", adjacentTriangleList)
    #np.save("out_" + str(meshSize)+ "_angleList", angleList)
    #np.save("out_" + str(meshSize)+ "_lineList", lineList)
    #
    print("Load mesh Over")
    pointValue = initPointValue(pointNum)
    field = initField(pointNum)
    # Local Solver Part
    #refList = [[0,0], [0,1], [1,0], [1,1]]
    refList = [[-100.0,-100.0], [-100.0,100.0], [100.0,-100.0], [100.,100.]]
    lastValue = pointValue.copy()
    MaxIte = 3
    print("------calc Start ------------")
    for i in range(MaxIte):
        for rerf in refList: # Run once
            ascentList = sortDistanceNode(rerf, pointList)
            descentList = ascentList.copy()
            descentList.reverse()
            disList = [ascentList, descentList]
            for ad in range(0,2,1):
                for i in range(0, pointNum, 1):
                    loaclPointIndex = disList[ad][i][0]
                    for j in range(0 , len(adjacentTriangleList[loaclPointIndex]), 1): #
                        localCellIndex = adjacentTriangleList[loaclPointIndex][j] # 
                        localCell = list(cellList[localCellIndex]) #
                        rangeIndex = (localCell.index(loaclPointIndex))
                        AList = [1,0,0]
                        BList = [2,2,1]
                        Aflag = AList[rangeIndex]
                        Bflag = BList[rangeIndex]
                        tA = pointValue[localCell[Aflag]]
                        tB = pointValue[localCell[Bflag]]
                        tC = pointValue[localCell[rangeIndex]]
                        a = lineList[localCellIndex][Aflag]
                        b = lineList[localCellIndex][Bflag]
                        c = lineList[localCellIndex][rangeIndex]
                        alpha = angleList[localCellIndex][Aflag]
                        beta = angleList[localCellIndex][Bflag]
                        gamma = angleList[localCellIndex][rangeIndex]
                        fC = field[loaclPointIndex]
                        if gamma > np.pi /2:
                            print("Error: Acute Angle")
                        else:
                            tC = localSolver(tA, tB, tC, a,b,c, alpha, beta, fC)
                            if tC < 0: print("Error: tC negative")
                            #if tC == 10000: print("Error: not update",i )
                            pointValue[localCell[rangeIndex]] = tC
                lastValue = pointValue
        print(npla.norm(np.array(pointValue) - np.array(lastValue)))
    print("-------------------------")
    erp = []
    for i in range(len(pointValue)):
        if pointValue[i] == 10000:
            print(i)
            pointValue[i] = 0
            erp.append(i)
    pointValue = list(pointValue)
    pointList = list(pointList)
    print(max(pointValue), min(pointValue))
    # Save as Vtk
    pv1 = {}
    pv1["u"] = list(pointValue)
    #mwrite = meshio.Mesh(mesh.points,mesh.cells).write("res.vtk")
    meshio.write_points_cells("res.vtk", mesh.points, mesh.cells, pv1)

    








