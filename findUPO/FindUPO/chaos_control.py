import numpy as np
import scipy.integrate as spi
from math import cos, pi, sin
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from ipywidgets import FloatProgress
from IPython.display import display
from ipywidgets import interact, interactive, interact_manual, fixed
import ipywidgets as widgets
from itertools import cycle
from sklearn.cluster import MeanShift
import csv as csv
import pandas as pd
from scipy.misc import derivative
from scipy.linalg import inv
import textwrap
'''
----------
UPO Finder
----------
'''

def ExportTimeSeries(filename,r0,n,pmapLst):
    '''
    Exports time-series to a csv file.
    filename: name of csv file in form 'file.csv'
    r0: [list] Intial point.
    n: [Integer] Number of points.
    pmapLst: [pmapLst(r0,n)] Function giving time-series.
    '''
    data = pmapLst(r0,n)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(data)

def Load_Cut(datafile, cutoff, n, maxpts = 200000):
    '''
    Loads csv data and calculates how far each point moves after a specified number of  applications of the map
    (appended at the end of each data point). Chooses points that move less than a specificied cut-off value. Returns array.
    filename: name of csv file in form 'file.csv'
    cutoff: maximum distance cutoff value [positive float]
    n: number of iterations of the map [positive integer]
    maxpts:  maximum number of points [positive integer] (optional)
    '''
    # Import datafile
    file = open(datafile, 'r')
    datareader = csv.reader(file,delimiter=',')
    data = []
    lmter = 0
    for row in datareader:
        if lmter < maxpts:
            data.append(row)
            lmter += 1
        else: break
    for i in range(0,len(data),1):
        for j in range(0,len(data[i]),1):
            data[i][j] = float(data[i][j])

    # Distance moved after n iterations
    def distmoved(i):
        arr = np.array(data[i]) - np.array(data[i+n])
        return np.inner(arr,arr)

    # Collects points that move less than cutoff
    under = []
    for i in range(0,len(data)-n,1):
        dist = distmoved(i)
        if dist < cutoff:
            newrow = []
            for j in range(len(data[i])):
                newrow.append(data[i][j])
            newrow.append(dist)
            under.append(newrow)

    return under


def Cut(data, cutoff, n, maxpts = 200000):
    '''
    Takes map trajectory and calculates how far each point moves after a specified number of  applications of the map
    (appended at the end of each data point). Chooses points that move less than a specificied cut-off value. Returns array.
    dara: Trajectory in form [[x1[0],...,x1[d-1]],...,[xn[0],...,xn[d-1]]] for a d dimensional space an n data points. [List or array]
    cutoff: maximum distance cutoff value [positive float]
    n: number of iterations of the map [positive integer]
    maxpts:  maximum number of points [positive integer] (optional)
    '''
    if type(data) is np.ndarray:
        data = data.tolist()

    # Distance moved after n iterations
    def distmoved(i):
        arr = np.array(data[i]) - np.array(data[i+n])
        return np.inner(arr,arr)

    # Collects points that move less than cutoff
    under = []
    for i in range(0,len(data)-n,1):
        dist = distmoved(i)
        if dist < cutoff:
            newrow = []
            for j in range(len(data[i])):
                newrow.append(data[i][j])
            newrow.append(dist)
            under.append(newrow)

    return under

def ClustPlt_func(data, cluster, coord1 , coord2 , point_size , plot_size, thres, band, thresscale = 1, bandscale = 1):
    '''
    Function for interactive widget ClustPlt.
    data: [d by n] array where there are n data points and data[i] = [r0,...,rd-1,dist(r)]  for all i where dist [i] is
          distance moved by point. Load_Cut outputs this form of array.
    cluster: [True/False] Decides whether to perform clustering or not.
    coord1/coord2: [integer 0 to d-1] co-ordinate axis to plot.
    pointsize: [Float] Size of points for matplotlib scatter.
    plotsize: [Float] Size of matplotlib plot.
    thres: [Float] Maximum distance moved for a point to be plotted.
    band: [Float] Bandwidth for the clustering algorithm - roughly proportional t0 the size of clusters it will find
    RETURNS: A plot of reccurring points with different colours for different clusters if clustering is True.
    '''
    # scale thres
    thres = thres*(1/thresscale)

    # scale bandwith
    band = band*(1/bandscale)

    # Type check
    coord1 = int(coord1)
    coord2 = int(coord2)

    def distmoved(i):
        return data[i][-1]

    # Collects points that move less than thres
    under = []
    for i in range(0,len(data),1):
        if distmoved(i) < thres:
            under.append(data[i][0:(len(data[i])-1)])


    if cluster == False:
        # Plots x and p values of points
        plt.figure(figsize=(plot_size,plot_size))
        plt.scatter(np.transpose(under)[coord1],np.transpose(under)[coord2],s=point_size)
        plt.show()
        print('Threshold: ', thres)
        print('Total number of points: ', len(under))

    if cluster == True:
        # Performs meanshift clustering
        ms = MeanShift(bandwidth=band, bin_seeding=True)
        ms.fit(under)
        labels = ms.labels_  #gives cluster label of datapoint in the same place of that datapoint in array
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        cluster_centers = ms.cluster_centers_

        # Plots clusters in differnent colors
        colors = cycle('grbcmykbgrcmykbgrcmykbgrcmyk')
        plt.figure(figsize=(plot_size,plot_size))
        for k, col in zip(range(n_clusters), colors): # gives (k=0,col=g),(k=1,col=r),...
            # collects all points of given cluster
            tmp = []
            for i in range(len(labels)):
                if labels[i] == k:
                    tmp.append(under[i])
            tmp = np.transpose(tmp)
            # plots points of cluster in given colour
            plt.scatter(tmp[coord1],tmp[coord2],color = col,s = point_size)
            #plots centre of cluster
            plt.scatter(cluster_centers[k][coord1],cluster_centers[k][coord2],color = col, marker = '+')
        plt.show()

        print('Total number of points: ', len(under))
        print('Number of clusters: ', n_clusters)
        print('Threshold: ', thres)
        print('Bandwidth: ', band)


def ClustPlt(data, thresopt = [0,2,0.01], bandopt = [0,1,0.001]):
    '''
    Widget for ClustPlt_func.
    data: [d + 1 by n] array where there are n data points and data[i] = [r0,...,rd-1,dist(r)]  for all i where dist [i] is
          distance moved by point. Load_Cut outputs this form of array.
    thresopt: (optional) [3 by 1 array] options for ClustPlt_func parameter thres in form [min, max, stepsize]
    bandopt: (optional) [3 by 1 array] options for ClustPlt_func parameter band in form [min, max, stepsize]
    '''
    sysdim = len(data[0]) - 1
    # scaling thres slider
    xt = thresopt[2]
    if xt < 0.01:
        x = xt
        nz = -1 # number of zeros between point and sig fig
        while x < 1:
            nz = nz + 1
            x = x*10
        thsc = 10**(nz - 1)  # 0.00a -> 0.0a, 0.000a -> 0.0a etc.
    else:
        thsc = 1
    # scaling band slider
    xb = bandopt[2]
    if xb < 0.01:
        x = xb
        nz = -1 # number of zeros between point and sig fig
        while x < 1:
            nz = nz + 1
            x = x*10
        bsc = 10**(nz - 1)  # 0.00a -> 0.0a, 0.000a -> 0.0a etc.
    else:
        bsc = 1


    interact(ClustPlt_func,
             data = fixed(data),
             cluster = False,
             coord1 =  widgets.IntSlider(min = 0, max = sysdim - 1, step = 1, value = 0),
             coord2 =  widgets.IntSlider(min = 0, max = sysdim - 1, step = 1, value = 1),
             plot_size = widgets.FloatSlider(min = 5, max = 15, step = 0.05, value = 7),
             point_size =  widgets.IntSlider(min = 1, max = 30, step = 1, value = 10),
             thres = widgets.FloatSlider(min = thresopt[0]*thsc, max = thresopt[1]*thsc, step = thresopt[2]*thsc, value = thsc*(thresopt[1] - thresopt[0])/2),
             band = widgets.FloatSlider(min = bandopt[0]*bsc, max = bandopt[1]*bsc, step = bandopt[2]*bsc, value = bsc*(bandopt[1] - bandopt[0])/2),
             thresscale = fixed(thsc),
             bandscale = fixed(bsc))

def FindUPO(data, n, thres, band, pmap, cluster = True,cutoff = 10**(-6),tooclose = 0.01,opttimes = 1):

    '''
    ------------------------------------------------------------------------------------------------
    FindUPO: Finds estimates for locations Unstable Periodic Orbits in phase space for chaotic maps.
    ------------------------------------------------------------------------------------------------
    The initial guesses are obtained by identifying clusters of reccurring points in phase space and
    choosing the one that moves the least. The initial guesses are then turned into accurate estimates
    using local optimization. The estimates are then organized into their orbits. The algorithm ensures
    that all the UPOs are distinct.

    data: [List exported by Load_Cut] See Load_Cut. This is the initial data that the guesses are obtained from.
    n: [Positive integer] See Load_cut. Highest period of UPO that we look for. The algorithm may also identify
       UPOs with length of the divisors of n. Must be the same n as was used in Load_Cut.
    thres: [Positive Float] Our initial cut-off for the distance moved by a point after n iterations of the map.
           To obtain a good value for thres, see ClustPlt.
    band: [Positive Float] Bandwidth for the clustering algorithm. Larger values correspond to larger clusters in general.
           To obtain a good value for thres, see ClustPlt.
    pmap: [function] map in question.
    cluster: [Boolean] (optional) If true clustering procedure is perfored, otherwise everypoint is used as a guess.
    cutoff: [Positive Float] (optional) The largest distance that a point can move in n iterations of a map to be considered a UPO.
    tooclose: [Positive Float] (optional) Farthest distance between two points to be considered seperate.
    optimes: [Positive Integer] (optional) Number of times local optimisation is performed. A value higher than 1 will
            increase the time taken to perform the optimisation but may increase accuracy of estimates and may increase
            the number of estimates achieved bu the algorithm, especially for higher period UPOs.

    RETURNS: Nested list of UPO periods and trajectory in form:

            UPOlist = [[n1,[r1_1,...,r1_n1]],
                                 ...
                       [nm,[rm_1,...,rm_nm]]]

            Where m is number of estimates,
                  ni is the period if the ith estimate,
                  ri_j is a d by 1 list containing the location of the jth point of the ith estimate.

    '''

    def pmapLst(r,n):
        out = [r]
        for i in range(n):
            out.append(pmap(out[-1]))
        return out

    '''
    Returns list of divisors of n
    n: [Positive integer]
    RETURNS: list
    '''
    def factor(n):
        out = []
        for i in range(1,n,1):
            if n%i ==0:
                out.append(i)
        return out

    '''
    Euclidean distance moved by a point r after n iterations of the map pmap(r,n).
    r: [d by 1 array] initial point
    n: number of iterations
    pmap: [function] map in question
    '''
    def dist(r,n, pmap=pmap):
        now = np.array(r)
        then = pmapLst(r,n)[-1]
        out = np.dot(now - then,now - then)
        return out

    # Apply threshold

    under = []
    for i in range(len(data)):
        if data[i][-1] < thres:
            under.append(data[i])

    # Obtaining initial guesses

    justpoints = []
    for i in range(len(under)):
        justpoints.append(under[i][0:(len(data[i])-1)])

    if cluster == False: # Take all points
        guesses = justpoints
    else: # Take point in cluster that moves the least
        guesses = []
        ms = MeanShift(bandwidth=band, bin_seeding=True)
        ms.fit(justpoints)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        for k in labels_unique:
            tmp = []
            for i in range(len(under)):
                if labels[i] == k:
                    tmp.append(under[i])
            ind = np.argmin(np.transpose(tmp)[-1])
            guesses.append(tmp[ind][0:(len(data[i])-1)])
    # Specific distance function

    def distn(r0):
        return dist(r0,n)

    # Improve guesses

    Impg = []
    ld = FloatProgress(min=0, max=len(guesses)-1)
    display(ld)
    for i in range(len(guesses)):
        ld.value = i
        tmp = guesses[i]
        # Optimising
        for j in range(opttimes):
            tmp = minimize(distn,tmp).x
        tmp = tmp.tolist()
        tmp.append(distn(tmp))
        Impg.append(tmp)

    # Are they good enough?

    goodUPOs = []
    for i in range(len(Impg)):
        if Impg[i][-1] < cutoff:
            goodUPOs.append(Impg[i][0:(len(data[i])-1)])

    # Is it length n or shorter?

    f = factor(n)

    UPOs = []
    globaldellist = []
    for i in f:
        def disttmp(r0):
            return dist(r0,i)
        dellist = []
        for j in range(len(goodUPOs)):
            #mintst = minimize(disttmp,goodUPOs[j]).x
            #movedby = np.dot(np.array(mintst) - np.array(goodUPOs[j]),np.array(mintst) - np.array(goodUPOs[j]))
            #if (disttmp(mintst) < cutoff and movedby < tooclose) or disttmp(goodUPOs[j]) < cutoff:
            if disttmp(goodUPOs[j]) < cutoff:
                dellist.append(j)
        for j in dellist:
            if j not in globaldellist:
                UPOs.append([i,goodUPOs[j]])
                globaldellist.append(j)
    for i in range(len(goodUPOs)):
        if i not in globaldellist:
            UPOs.append([n,goodUPOs[i]])

    # Current structure:
    # UPOs = [[n1,[r1[0],...,r1[4]]],
    #              ...
    #         [nm,[rm[0],...,rm[4]]]]

    # Are there two in the same place?

    closelist = [] # List of indice pairs [i,j] such that the ith and ith UPOs are too close
    for i in range(len(UPOs)):
        if i < len(UPOs) - 1:
            for j in range(i + 1,len(UPOs),1):
                diff = np.array(UPOs[i][1]) - np.array(UPOs[j][1])
                if np.dot(diff,diff) < tooclose and UPOs[i][0] == UPOs[j][0]:
                    closelist.append([i,j])
    mergedlist = [] # List of all indices grouped together if their corresponding data points are too close

    for i1 in range(len(UPOs)):
        added = 0
        related = [] # List of indices related to i1
        for j1 in range(len(closelist)):
            for j2 in range(len(closelist[j1])):
                if closelist[j1][j2] == i1:
                    j3 = (j2 + 1)%2
                    related.append(closelist[j1][j3])
        for i2 in range(len(mergedlist)):
            for i3 in range(len(mergedlist[i2])):
                for i4 in range(len(related)):
                    if related[i4] == mergedlist[i2][i3] and added == 0:
                        mergedlist[i2].append(i1)
                        added = 1
        if added == 0:
            mergedlist.append([i1])

    UPOstmp = []
    for i in range(len(mergedlist)): # Choosing best point of all groups of similar UPOs
        minind = mergedlist[i][0]
        for j in range(1,len(mergedlist[i])):
            if dist(UPOs[mergedlist[i][j]][1],UPOs[mergedlist[i][j]][0]) < dist(UPOs[minind][1],UPOs[minind][0]):
                minind = mergedlist[i][j]
        UPOstmp.append(UPOs[minind])

    UPOs = UPOstmp


    # Linking Trajectories

    def closecheck(UPOs):
        closelist = [] # List of indice pairs [i,j] such that the ith and jth UPOs are too close
        for i in range(len(UPOs)):
            if i < len(UPOs) - 1:
                for j in range(i + 1,len(UPOs),1):
                    if UPOs[i][0] == UPOs[j][0]:
                        same = 0
                        for k1 in range(len(UPOs[i][1])):
                            for k2  in range(len(UPOs[j][1])):
                                diff = np.array(UPOs[i][1][k1]) - np.array(UPOs[j][1][k2])
                                if np.dot(diff,diff) < tooclose:
                                    same = 1
                        if same == 1:
                            closelist.append([i,j])

        mergedlist = [] # List of all indices grouped together if their corresponding data points are too close
        for i1 in range(len(UPOs)):
            added = 0
            related = [] # List of indices related to i1
            for j1 in range(len(closelist)):
                for j2 in range(len(closelist[j1])):
                    if closelist[j1][j2] == i1:
                        j3 = (j2 + 1)%2
                        related.append(closelist[j1][j3])
            for i2 in range(len(mergedlist)):
                for i3 in range(len(mergedlist[i2])):
                    for i4 in range(len(related)):
                        if related[i4] == mergedlist[i2][i3] and added == 0:
                            mergedlist[i2].append(i1)
                            added = 1
            if added == 0:
                mergedlist.append([i1])

        UPOstmp = []

        for i in range(len(mergedlist)): # Choosing best point of all groups of similar UPOs
            minind = mergedlist[i][0]
            for j in range(1,len(mergedlist[i])):
                if dist(UPOs[mergedlist[i][j]][1][0],UPOs[mergedlist[i][j]][0]) < dist(UPOs[minind][1][0],UPOs[minind][0]):
                    minind = mergedlist[i][j]
            UPOstmp.append(UPOs[minind])
        return UPOstmp

    UPOstmp = []
    for i in range(len(UPOs)):
        orbit = pmapLst(UPOs[i][1],UPOs[i][0] - 1) # full orbit
        orbitlist = [] # converted to list
        for j in range(len(orbit)):
            orbitlist.append(np.array(orbit[j]).tolist())
        UPOstmp.append([UPOs[i][0],orbitlist])
    UPOs = UPOstmp

    UPOs = closecheck(UPOs)
    return UPOs


def appendUPOlist(UPOlist1, UPOlist2, pmap, tooclose = 0.01):
    '''
    Combines lists of UPOs exported by FindUPOs, checking that no two are the same.
    UPOlist1/UPOlist2: [Nested lists] UPO lists
    dist: [Float function of state space] (optional) Distance function used.
    tooclose: [Float] (optional) Farthest distance between two points to be considered seperate.
    '''
    # Defining distance function
    def pmapLst(r,n):
        out = [r]
        for i in range(n):
            out.append(pmap(out[-1]))
        return out

    '''
    Euclidean distance moved by a point r after n iterations of the map pmap(r,n).
    r: [d by 1 array] initial point
    n: number of iterations
    pmap: [function] map in question
    '''
    def dist(r,n, pmap=pmap):
        now = np.array(r)
        then = pmapLst(r,n)[-1]
        out = np.dot(now - then,now - then)
        return out


    # Combining
    UPOlist = []
    for i in range(len(UPOlist1)):
        UPOlist.append(UPOlist1[i])
    for i in range(len(UPOlist2)):
        UPOlist.append(UPOlist2[i])

    def closecheck(UPOs):
        closelist = [] # List of indice pairs [i,j] such that the ith and ith UPOs are too close
        for i in range(len(UPOs)):
            if i < len(UPOs) - 1:
                for j in range(i + 1,len(UPOs),1):
                    if UPOs[i][0] == UPOs[j][0]:
                        same = 0
                        for k1 in range(len(UPOs[i][1])):
                            for k2  in range(len(UPOs[j][1])):
                                diff = np.array(UPOs[i][1][k1]) - np.array(UPOs[j][1][k2])
                                if np.dot(diff,diff) < tooclose:
                                    same = 1
                        if same == 1:
                            closelist.append([i,j])

        mergedlist = [] # List of all indices grouped together if their corresponding data points are too close
        for i1 in range(len(UPOs)):
            added = 0
            related = [] # List of indices related to i1
            for j1 in range(len(closelist)):
                for j2 in range(len(closelist[j1])):
                    if closelist[j1][j2] == i1:
                        j3 = (j2 + 1)%2
                        related.append(closelist[j1][j3])
            for i2 in range(len(mergedlist)):
                for i3 in range(len(mergedlist[i2])):
                    for i4 in range(len(related)):
                        if related[i4] == mergedlist[i2][i3] and added == 0:
                            mergedlist[i2].append(i1)
                            added = 1
            if added == 0:
                mergedlist.append([i1])

        UPOstmp = []

        for i in range(len(mergedlist)): # Choosing best point of all groups of similar UPOs
            minind = mergedlist[i][0]
            for j in range(1,len(mergedlist[i])):
                if dist(UPOs[mergedlist[i][j]][1][0],UPOs[mergedlist[i][j]][0]) < dist(UPOs[minind][1][0],UPOs[minind][0]):
                    minind = mergedlist[i][j]
            UPOstmp.append(UPOs[minind])
        return UPOstmp

    return closecheck(UPOlist)


def saveUPOlist(UPOlist,filename):
    '''
    Saves nested list, such a the UPO list exported by FindUPO to a pickle file.
    UPOlist: [Nested list] List of UPOs such as the ine exported by FindUPO
    filename: Name of file in the form: 'filename.p'
    '''
    df = pd.DataFrame(UPOlist)
    df.to_pickle(filename)


def loadUPOlist(filename):
    '''
    Loads nested list, such a the UPO list exported by FindUPO from a pickle file.
    filename: Name of file in the form: 'filename.p'
    '''
    df = pd.read_pickle(filename)
    dflist = df.values.tolist()
    return dflist


def toMathematica(UPOlist,printer = True, filename = False ):
    '''
    Converts any python list to a Mathematica list and prints it and/or daves it to
    a text file.
    UPOlist: [list/array] List you wish to convert.
    printer: (optional) [boolean] Pass True/False to turn printing on/off.
    filename: (optional) [string] Name of file in form 'filename.txt'
    '''
    out = textwrap.wrap(str(UPOlist).replace('[','{').replace(']','}').replace('e','*10^'),100)
    if filename != False:
        with open(filename, "w") as text_file:
            for row in out:
                print(row, file=text_file)
    if printer == True:
        print(UPOlist)
'''
---------------
Jacobian Finder
---------------
'''
def FindJac(r,pmap,n=1):

    '''
    Finds Jacobian of the nth iterate of some map evaluated at a point r.
    https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
    r: [list/array] Point to evaluate at.
    pmap: [pmap(r)] Map in question.
    n: [pos. integer] Number of iterates of the map.
    '''

    dd = len(r) # domain dimension
    di = len(pmap(r)) # image dimension

    def f(r): # nth iterate of pmap
        out = r
        for i in range(n):
            out = pmap(out)
        return out

    Jac = []
    for i in range(di):
        row = []
        for j in range(dd):
            def fi(xj): # ith component of f with all but jth input fixed at r
                rtmp = [] # r with jth component freed
                rtst = []
                for k in range(j):
                    rtmp.append(r[k])
                rtmp.append(xj)
                for k in range(j+1,di):
                    rtmp.append(r[k])
                return f(rtmp)[i]
            row.append(derivative(fi,r[j],order = 5))
        Jac.append(row)

    return Jac

def FindB(r,pmapb,beval):

    '''
    Finds the derivative of a map with respect to a parameter.
    r: [list/array] Point to evaluate map at.
    pmap: [pmap(r,b=#)] Map in question.
    b: [parameter] Parameter in quesion.
    beval: [float] Value of parameter to evaluate derivative.
    '''

    d = len(pmapb(r,beval))

    out = []
    for i in range(d):
        def fi(bv):
            return pmapb(r,bv)[i]
        out.append(derivative(fi,beval))

    return out

def JUPOB(UPOlist,pmap,pmapb,beval):
    '''
    Given a UPOlist (see FindUPO), appends a list of Jacobians and B matrices
    (derivative of map with respect to the control parameter) evaluated at each
    point of the UPO. A row of the UPOlist (i.e. all the info for a single UPO)
    is given back as,
    UPOlist[i] = [n,[r1,...,rn],[Df(r1),...,Df(rn)],[B(r1),...,B(rn)]]
    UPOlist: [nested list] List of periodic orbits in format given by FindUPO.
    pmap: [pmap(r)] Map in question.
    pmapb: [pmap(r,b)] Map in question with control parameter passed as second
        parameter.
    beval: [float] Non-control value for control parameter.
    '''
    for i in range(len(UPOlist)):
        JLst = []
        BLst = []
        for j in range(len(UPOlist[i][1])):
            JLst.append(FindJac(r = UPOlist[i][1][j], pmap = pmap))
            BLst.append(FindB(r = UPOlist[i][1][j], pmapb = pmapb, beval = beval))
        UPOlist[i].append(JLst)
        UPOlist[i].append(BLst)
    return UPOlist

def JUPO(UPOlist,pmap):
    '''
    Given a UPOlist (see FindUPO), appends a list of Jacobians evaluated at each
    point of the UPO. A row of the UPOlist (i.e. all the info for a single UPO)
    is given back as,
    UPOlist[i] = [n,[r1,...,rn],[Df(r1),...,Df(rn)]]
    UPOlist: [nested list] List of periodic orbits in format given by FindUPO.
    pmap: [pmap(r)] Map in question.
    '''
    for i in range(len(UPOlist)):
        JLst = []
        for j in range(len(UPOlist[i][1])):
            JLst.append(FindJac(r = UPOlist[i][1][j], pmap = pmap))
        UPOlist[i].append(JLst)
    return UPOlist

'''
-------
Control
-------
'''

def acker(A,B,s):
    '''
    Ackermann's formula,
    https://en.wikipedia.org/wiki/Ackermann%27s_formula
    A:[matrix] Jacobian
    B:[list/array] Derivative with respect to control parameter
    s:[list] list of desired eigenvalues.
    '''

    if len(A) != len(np.transpose(A)):
        print('Jacobian must be square.')

    n = len(A) # Dimension of system

    if len(s) != n:
        print('Number of desired eigevalues must be equal to the dimension of the system.')

    A = np.array(A)
    B = np.array(B)

    O = np.zeros(n)
    O[-1] = 1

    Ginv = []
    for i in range(n):
        if i == 0:
            Ginv.append(B)
        else:
            An = A
            for j in range(i-1):
                An = np.dot(An,A)
            Ginv.append(np.dot(An,B))
    Ginv = np.transpose(Ginv)
    G = inv(Ginv) # Controllabillity matrix

    I = np.identity(n)
    Y = I
    for i in range(n): # Characteristic ploynomial
        Y = np.dot(Y,A - s[i]*I)

    OGY = np.dot(O,np.dot(G,Y))

    return OGY



'''
----
Maps
----
'''

def henon(r,a = 1.4,b = 0.3):
    return [1 - a*r[0]**2 + r[1], b*r[0]]

def henonLst(r0,n):
    out = [r0]
    for i in range(n):
        out.append(henon(out[-1]))
    return out

def stmap(r,K=5.6):
    pnxt = r[1] + K*sin(r[0])
    xnxt = r[0] + pnxt
    return [xnxt%(2*pi),pnxt%(2*pi)]

def stmapLst(r0,n):
    out = [r0]
    for i in range(n):
        out.append(stmap(out[-1]))
    return out

def stmapnb(r,K=5.6):
    pnxt = r[1] + K*sin(r[0])
    xnxt = r[0] + pnxt
    return [xnxt,pnxt]

from scipy.integrate import odeint
'''
Poincare return map for average Semiclassical Duffing
r0: Initial condition [5 by 1 list\array]
RETURNS: [5 by 1 array]
'''
def SnDuffP(r0):
    #Parameters
    Gamma = 0.1
    g = 0.3
    omega = 1
    beta = 0.2
    theta = 0
    dens = 3000
    u = 1
    u1 = u*cos(theta)
    u2 = u*sin(theta)

    def f(r,t):
        x = r[0]
        p = r[1]
        sx = r[2]
        sp = r[3]
        sxp = r[4]

        dxdt = p
        dpdt = -(beta**2)*(x**3 + 3*sx*x) + x - 2*Gamma*p + (g/beta)*cos(omega*t)
        dsxdt = 2.0*sxp + 0.5*Gamma*(1.0 - 4.0*(sxp**2 - sx + sx**2) +u1*(-1.0+4.0*((sxp**2) + sx - sx**2))+4.0*u2*(-sxp+2.0*sxp*sx))
        dspdt = 2.0*sxp - 6.0*(beta**2)*sxp*(x**2+sx) + 0.5*Gamma*(1.0-4.0*sp-4.0*(sp**2)-4.0*sxp**2 + u1*(1.0-4.0*sp+4.0*(sp**2)-4.0*sxp**2)+4.0*u2*(-sxp+2.0*sxp*sp))
        dsxpdt = -2.0*Gamma*(sxp*(sp+sx) - u1*sxp*(sp-sx)-0.25*u2*(1.0-2.0*sp+4.0*(sxp**2)-2.0*sx+4.0*sp*sx)) + sp +sx -3.0*(beta**2)*sx*(x**2+sx)
        drdt = np.array([dxdt,dpdt, dsxdt, dspdt, dsxpdt])

        return drdt
    r0 = np.array([r0[0],r0[1],r0[2],r0[3],r0[4]])
    out = odeint(f,r0,np.linspace(0,2*pi/omega,dens))[-1]
    return out

'''
Poincare return map list for average Semiclassical Duffing (using SnDuffP)
r0: [5 by 1 array] Initial condition
n: [positive integer] Number of points
'''
def SnDuffPLst(r0,n):
    out = [r0]
    for i in range(0,n,1):
        out.append(SnDuffP(out[-1]))
    return out

# Poincare section (stroboscopic) for Duffing oscillator.
def DuffP(r0,b = 0.125, g = 3, omega = 1):

    def f(r,t):

        x = r[0]
        y = r[1]
        drdt = np.array([ y, - 2*b*y + x - x**3 + g*cos(omega*t)])
        return drdt

    tvals = np.linspace(0,2*pi,3)
    temp = odeint(f,r0,tvals)[-1]
    return temp

def DuffPLst(r0,n):
    out = [r0]
    for i in range(0,n,1):
        out.append(DuffP(out[-1]))
    return out
