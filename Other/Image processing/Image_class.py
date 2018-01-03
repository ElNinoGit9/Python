class image_class:
    def __init__(self, source):
        import numpy as np
        from scipy import misc
        from scipy import ndimage
        import matplotlib.pyplot as plt
        import math
        import random
        '''f = misc.face(gray=True)'''

        f = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\Sudoku00.png')
        # 255 = White
        # 0 = Black
        lx, ly, lz = f.shape
        print(lx, ly, lz)
        print(f)
        f[:,:,0] = (f[:,:,0] > 220)*255
        f[:,:,1] = (f[:,:,1] > 220)*255
        f[:,:,2] = (f[:,:,2] > 220)*255
        #f[:,:,3] = 255
        plt.imshow(f)
        plt.show()
        #Find line
        L  = np.zeros((ly, 1))
        L2 = np.zeros((ly, 1))
        R  = np.zeros((lx, 1))
        R2 = np.zeros((lx, 1))

        for k in range(0,ly):
            L[k] = (np.sum(f[:, k, 0]) + np.sum(f[:, k, 1]) + np.sum(f[:, k, 2]))/len(L2)
            L2[k] = math.pow(np.sum(f[:, k, 0]) + np.sum(f[:, k, 1]) + np.sum(f[:, k, 2]), 2)

        for l in range(0, lx):
            R[l] = (np.sum(f[l, :, 0]) + np.sum(f[l, :, 1]) + np.sum(f[l, :, 2]))/len(R2)
            R2[l] = math.pow(np.sum(f[l, :, 0]) + np.sum(f[l, :, 1]) + np.sum(f[l, :, 2]), 2)

        VarL = L2/len(L2) - np.square(L)/len(L)
        VarR = R2/len(R2) - np.square(R)/len(R)
        stdL = np.sqrt(VarL)
        stdR = np.sqrt(VarR)

        for k in range(0,ly):
            if stdL[k] < 1000:
                f[:,k,0] = 0
                f[:,k,1] = 255
                f[:,k,2] = 0

        for l in range(0, lx):
            if stdR[l] < 1000:
                f[l,:,0] = 0
                f[l,:,1] = 255
                f[l,:,2] = 0
        '''
        for k in range(0,ly):
            if stdL[k] < 100:
                f[:,k,0] = 255
                f[:,k,1] = 255
                f[:,k,2] = 255

        for l in range(0, lx):
            if stdR[l] < 100:
                f[l,:,0] = 255
                f[l,:,1] = 255
                f[l,:,2] = 255
        '''
        # find squares
        squares = 9
        M = np.zeros((squares, 4))
        tmprow = 0
        tmpcol = 0
        tmprowEnd = 0
        tmpcolEnd = 0

        '''
        for k in range(1, ly):

            if (((f[round(lx/(squares*2)), ly - k, 0] == 0) & \
                (f[round(lx/(squares*2)), ly - k, 1] == 0) & \
                (f[round(lx/(squares*2)), ly - k, 2] == 255)) & \
                ((f[round(lx/(squares*2)), ly - k - 1, 0] != 0) | \
                (f[round(lx/(squares*2)), ly - k - 1, 1] != 0) | \
                (f[round(lx/(squares*2)), ly - k - 1, 2] != 255))) & (tmpcolEnd < squares):

                M[squares - 1 - tmpcolEnd, 3] = ly-k
                tmpcolEnd = tmpcolEnd + 1


            if ((f[round(lx/(squares*2)), k-1, 0] == 0) & \
                (f[round(lx/(squares*2)), k-1, 1] == 0) & \
                (f[round(lx/(squares*2)), k-1, 2] == 255)) & \
                ((f[round(lx/(squares*2)), k, 0] != 0) | \
                (f[round(lx/(squares*2)), k, 1] != 0) | \
                (f[round(lx/(squares*2)), k, 2] != 255)) & (tmpcol < squares):

                M[tmpcol, 2] = k
                tmpcol = tmpcol + 1
                #print('hej')

        for l in range(1, lx):

            if (((f[lx - l, round(ly/(squares*2)), 0] == 0) & \
                (f[lx - l, round(ly/(squares*2)), 1] == 0) & \
                (f[lx - l, round(ly/(squares*2)), 2] == 255)) & \
                ((f[lx - l - 1, round(ly/(squares*2)), 0] != 0) | \
                (f[lx - l - 1, round(ly/(squares*2)), 1] != 0) | \
                (f[lx - l - 1, round(ly/(squares*2)), 2] != 255)) & (tmprowEnd < squares)):

                M[squares - tmprowEnd - 1, 1] = lx - l
                tmprowEnd = tmprowEnd + 1

            if (((f[l-1, round(ly/(squares*2)), 0] == 0) & \
                (f[l-1, round(ly/(squares*2)), 1] == 0) & \
                (f[l-1, round(ly/(squares*2)), 2] == 255)) & \
                ((f[l, round(ly/(squares*2)), 0] != 0) | \
                (f[l, round(ly/(squares*2)), 1] != 0) | \
                (f[l, round(ly/(squares*2)), 2] != 255)) & (tmprow < squares)):

                M[tmprow, 0] = l
                tmprow = tmprow + 1
        '''
        print(f)
        for k in range(1, ly):

            if (((f[round(lx/(squares*2)), ly - k, 0] == 0) & \
                (f[round(lx/(squares*2)), ly - k, 1] == 255) & \
                (f[round(lx/(squares*2)), ly - k, 2] == 0)) & \
                ((f[round(lx/(squares*2)), ly - k - 1, 0] == 255) & \
                (f[round(lx/(squares*2)), ly - k - 1, 1] == 255) & \
                (f[round(lx/(squares*2)), ly - k - 1, 2] == 255))) & (tmpcolEnd < squares):
                M[squares - 1 - tmpcolEnd, 3] = ly-k
                tmpcolEnd = tmpcolEnd + 1

                #f[:,ly-k,0] = 255
                #f[:,ly-k,1] = 0
                #f[:,ly-k,2] = 0

            if ((f[round(lx/(squares*2)), k-1, 0] == 0) & \
                (f[round(lx/(squares*2)), k-1, 1] == 255) & \
                (f[round(lx/(squares*2)), k-1, 2] == 0)) & \
                ((f[round(lx/(squares*2)), k, 0] == 255) & \
                (f[round(lx/(squares*2)), k, 1] == 255) & \
                (f[round(lx/(squares*2)), k, 2] == 255)) & (tmpcol < squares):

                M[tmpcol, 2] = k
                tmpcol = tmpcol + 1
                #f[:,k,0] = 100
                #f[:,k,1] = 0
                #f[:,k,2] = 100
        for l in range(1, lx):

            if (((f[lx - l, round(ly/(squares*2)), 0] == 0) & \
                (f[lx - l, round(ly/(squares*2)), 1] == 255) & \
                (f[lx - l, round(ly/(squares*2)), 2] == 0)) & \
                ((f[lx - l - 1, round(ly/(squares*2)), 0] == 255) & \
                (f[lx - l - 1, round(ly/(squares*2)), 1] == 255) & \
                (f[lx - l - 1, round(ly/(squares*2)), 2] == 255)) & (tmprowEnd < squares)):

                M[squares - tmprowEnd - 1, 1] = lx - l
                tmprowEnd = tmprowEnd + 1
                #f[lx-l,:,0] = 0
                #f[lx-l,:,1] = 0
                #f[lx-l,:,2] = 255

            if (((f[l-1, round(ly/(squares*2)), 0] == 0) & \
                (f[l-1, round(ly/(squares*2)), 1] == 255) & \
                (f[l-1, round(ly/(squares*2)), 2] == 0)) & \
                ((f[l, round(ly/(squares*2)), 0] == 255) & \
                (f[l, round(ly/(squares*2)), 1] == 255) & \
                (f[l, round(ly/(squares*2)), 2] == 255)) & (tmprow < squares)):

                M[tmprow, 0] = l
                tmprow = tmprow + 1
                #f[l,:,0] = 0
                #f[l,:,1] = 100
                #f[l,:,2] = 100

        plt.imshow(f)
        plt.show()
        # Find which squares including numbers

        S_num = np.zeros((9, 9))
        default_num = 36 #27 #30 #35
        def_size = 30
        Numbers = np.zeros((def_size, def_size, default_num))
        tmp = 0
        tmp2 = 0
        indices = np.zeros((81, 2))
        empt_indices = np.zeros((81, 2))
        n_min = np.zeros((9,9,2))
        n_max = np.zeros((9,9,2))
        n_mean = np.zeros((9,9,2))
        print(M)

        #f[M[:,0], :, 0] = 0
        #f[M[:,0], :, 1] = 0
        #f[M[:,0], :, 2] = 255

        for k in range(0, 9):

            for l in range(0, 9):
                #if (((np.max(f[M[l,0]:M[l,1], M[k,2]:M[k,3], 0]) - np.min(f[M[l,0]:M[l,1], M[k,2]:M[k,3], 0])) > 100) | \
                #   ((np.max(f[M[l,0]:M[l,1], M[k,2]:M[k,3], 1]) - np.min(f[M[l,0]:M[l,1], M[k,2]:M[k,3], 1])) > 100) | \
                #   ((np.max(f[M[l,0]:M[l,1], M[k,2]:M[k,3], 2]) - np.min(f[M[l,0]:M[l,1], M[k,2]:M[k,3], 2])) > 100)):
                if (np.mean(f[M[l,0]:M[l,1], M[k,2]:M[k,3], 0]) < 250):
                       S_num[l, k] = 10
                       indices[tmp, :] = [l, k]
                       n_min[l, k, :] = np.min(np.where(f[M[l,0]:M[l,1], M[k,2]:M[k,3], 0] < 10), 1)
                       n_max[l, k, :] = np.max(np.where(f[M[l,0]:M[l,1], M[k,2]:M[k,3], 0] < 10), 1)
                       n_mean[l, k, :] = np.round((n_max[l,k,:] + n_min[l,k,:])/2.0)

                       Numbers[:,:,tmp] = f[M[l,0] + n_mean[l,k,0] - def_size/2.0:M[l,0] + n_mean[l,k,0] + def_size/2.0, M[k,2] + n_mean[l,k,1] - def_size/2.0:M[k,2] + n_mean[l,k,1] + def_size/2.0, 1]

                       if (n_mean[l,k,0] - def_size/2.0 < 0):
                           shiftrow = 1
                       else:
                           shiftrow = 0
                       if (n_mean[l,k,1] - def_size/2.0 < 0):
                           shiftcol = 1
                       else:
                           shiftcol = 0

                       n_min[l, k, :] = np.min(np.where(f[M[l,0] + n_mean[l,k,0] - def_size/2.0 + shiftrow:M[l,0] + n_mean[l,k,0] + def_size/2.0, M[k,2] + n_mean[l,k,1] - def_size/2.0 + shiftcol :M[k,2] + n_mean[l,k,1] + def_size/2.0, 2] < 100), 1)
                       n_max[l, k, :] = np.max(np.where(f[M[l,0] + n_mean[l,k,0] - def_size/2.0 + shiftrow:M[l,0] + n_mean[l,k,0] + def_size/2.0, M[k,2] + n_mean[l,k,1] - def_size/2.0 + shiftcol :M[k,2] + n_mean[l,k,1] + def_size/2.0, 2] < 100), 1)

                       tmp = tmp + 1

                else:
                   n_min[l, k, :] = np.min(np.where(f[M[l,0]:M[l,1], M[k,2] :M[k,3], 0] > 100), 1)
                   n_max[l, k, :] = np.max(np.where(f[M[l,0]:M[l,1], M[k,2] :M[k,3], 0] > 100), 1)
                   n_mean[l, k, :] = np.round((n_max[l, k, :] + n_min[l, k, :])/2.0)
                   empt_indices[tmp2, :] = [l, k]
                   tmp2 = tmp2 + 1

        print(S_num)
        # Cut out numbers

        RealNumbers = np.zeros((30,30,9))


        min_num = np.zeros((9,2))
        max_num = np.zeros((9,2))
        mean_num = np.zeros((9,2))
        min_num[:,0] = [5,6,6,6,6,6,6,6,6]
        min_num[:,1] = [11,8,8,8,8,8,8,8,8]
        max_num[:,0] = [23,25,25,25,25,25,25,25,25]
        max_num[:,1] = [20,22,22,22,22,22,22,22,22]
        mean_num[:,0] = [24,24,24,24,24,24,24,24,24]
        mean_num[:,1] = [25,26,26,27,27,27,26,26,26]

        '''
        RealNumbers[:,0] = np.reshape(f[M[0,0] + mean_num[0,0] - 15:M[0,0] + mean_num[0,0] + 15, M[0,2] + mean_num[0,1] - 15:M[0,2] + mean_num[0,1] + 15, 2], (30*30,))
        RealNumbers[:,1] = np.reshape(f[M[1,0] + mean_num[1,0] - 15:M[1,0] + mean_num[1,0] + 15, M[4,2] + mean_num[1,1] - 15:M[4,2] + mean_num[1,1] + 15, 2], (30*30,))
        RealNumbers[:,2] = np.reshape(f[M[2,0] + mean_num[2,0] - 15:M[2,0] + mean_num[2,0] + 15, M[8,2] + mean_num[2,1] - 15:M[8,2] + mean_num[2,1] + 15, 2], (30*30,))
        RealNumbers[:,3] = np.reshape(f[M[2,0] + mean_num[3,0] - 15:M[2,0] + mean_num[3,0] + 15, M[3,2] + mean_num[3,1] - 15:M[3,2] + mean_num[3,1] + 15, 2], (30*30,))
        RealNumbers[:,4] = np.reshape(f[M[2,0] + mean_num[4,0] - 15:M[2,0] + mean_num[4,0] + 15, M[4,2] + mean_num[4,1] - 15:M[4,2] + mean_num[4,1] + 15, 2], (30*30,))
        RealNumbers[:,5] = np.reshape(f[M[2,0] + mean_num[5,0] - 15:M[2,0] + mean_num[5,0] + 15, M[5,2] + mean_num[5,1] - 15:M[5,2] + mean_num[5,1] + 15, 2], (30*30,))
        RealNumbers[:,6] = np.reshape(f[M[3,0] + mean_num[6,0] - 15:M[3,0] + mean_num[6,0] + 15, M[5,2] + mean_num[6,1] - 15:M[5,2] + mean_num[6,1] + 15, 2], (30*30,))
        RealNumbers[:,7] = np.reshape(f[M[2,0] + mean_num[7,0] - 15:M[2,0] + mean_num[7,0] + 15, M[1,2] + mean_num[7,1] - 15:M[1,2] + mean_num[7,1] + 15, 2], (30*30,))
        RealNumbers[:,8] = np.reshape(f[M[2,0] + mean_num[8,0] - 15:M[2,0] + mean_num[8,0] + 15, M[2,2] + mean_num[8,1] - 15:M[2,2] + mean_num[8,1] + 15, 2], (30*30,))
        '''

        RealNumbers[:,:,0] = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\One.png')
        RealNumbers[:,:,1] = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\Two.png')
        RealNumbers[:,:,2] = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\Three.png')
        RealNumbers[:,:,3] = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\Four.png')
        RealNumbers[:,:,4] = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\Five.png')
        RealNumbers[:,:,5] = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\Six.png')
        RealNumbers[:,:,6] = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\Seven.png')
        RealNumbers[:,:,7] = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\Nine.png')
        RealNumbers[:,:,8] = misc.imread('C:\Users\Markus\Documents\Python Scripts\Image processing\Eight.png')

        # Interpolation
        RealNumbers_scaled = np.zeros((def_size,def_size,9))

        for k in range(0,9):

            dx = float(def_size)/(max_num[k,0] - min_num[k,0] + 1)
            RealNumbers_scaled[0,:,k]  = RealNumbers[min_num[k,0],:,k]
            RealNumbers_scaled[def_size-1,:,k] = RealNumbers[max_num[k,0],:,k]

            for l in range(1,def_size-1):

                div = math.floor(l/dx)
                rest = l/dx - math.floor(l/dx)
                RealNumbers_scaled[l,:,k] = (1-rest)*RealNumbers[min_num[k,0] + div,:,k] + (rest)*RealNumbers[min_num[k,0] + div+1,:,k]

        tmpScaled = np.zeros((def_size,def_size,9))
        tmpScaled[:,:,:] = RealNumbers_scaled[:,:,:]

        for k in range(0,9):

            dx = float(def_size)/(max_num[k,1] - min_num[k,1] + 1)
            RealNumbers_scaled[:,0,k]  = tmpScaled[:,min_num[k,1],k]
            RealNumbers_scaled[:,def_size-1,k] = tmpScaled[:,max_num[k,1],k]

            for l in range(1,def_size-1):
                div = math.floor(l/dx)
                rest = l/dx - math.floor(l/dx)
                RealNumbers_scaled[:,l,k] = (1-rest)*tmpScaled[:,min_num[k,1] + div,k] + (rest)*tmpScaled[:,min_num[k,1] + div + 1,k]

        # Interpolate Numbers

        Numbers_scaled = np.zeros((30,30,default_num))

        for k in range(0,default_num):
            [l1, k1] = indices[k, :]
            dx = 30.0/(n_max[l1,k1,0] - n_min[l1,k1,0] + 1)
            Numbers_scaled[0,:,k]  = Numbers[n_min[l1,k1,0],:,k]
            Numbers_scaled[29,:,k] = Numbers[n_max[l1,k1,0],:,k]

            for l in range(1,29):

                div = math.floor(l/dx)
                rest = l/dx - math.floor(l/dx)
                Numbers_scaled[l,:,k] = (1-rest)*Numbers[n_min[l1,k1,0] + div,:,k] + (rest)*Numbers[n_min[l1,k1,0] + div+1,:,k]

        tmpScaled = np.zeros((30,30,default_num))
        tmpScaled[:,:,:] = Numbers_scaled[:,:,:]

        for k in range(0,default_num):
            [l1, k1] = indices[k, :]
            dx = 30.0/(n_max[l1,k1,1] - n_min[l1,k1,1] + 1)
            Numbers_scaled[:,0,k]  = tmpScaled[:,n_min[l1,k1,1],k]
            Numbers_scaled[:,29,k] = tmpScaled[:,n_max[l1,k1,1],k]

            for l in range(1,29):
                div = math.floor(l/dx)
                rest = l/dx - math.floor(l/dx)
                Numbers_scaled[:,l,k] = (1-rest)*tmpScaled[:,n_min[l1,k1,1] + div,k] + (rest)*tmpScaled[:,n_min[l1,k1,1] + div + 1,k]

        # Classify numbers
        dist = np.zeros((9,1))
        ind = np.zeros((default_num,1))
        min_dist = 10*255*255*30*30*np.ones((9, 1))
        best_numbers = np.zeros((30,30,9))
        best_num_actual = np.zeros((30,30,9))

        for k in range(0,default_num): #len(ind)

            dist = np.zeros((9,1))

            for l in range(0,9):

                dist[l] = np.sum(np.square(np.reshape(Numbers_scaled[:,:,k], (30*30,)) - np.reshape(RealNumbers_scaled[:,:,l], (30*30,))))

                if dist[l] < min_dist[l]:
                    min_dist[l] = dist[l]
                    best_numbers[:,:,l] = Numbers_scaled[:,:,k]
                    best_num_actual[:,:,l] = Numbers[:,:,k]

        # Run again with best numbers
        for k in range(0,default_num): #len(ind)

            dist = np.zeros((9,1))

            for l in range(0,9):

                dist[l] = np.sum(np.square(np.reshape(Numbers_scaled[:,:,k], (30*30,)) - np.reshape(best_numbers[:,:,l], (30*30,))))

            ind[k] = np.argmin(dist)
        # Create sudoku matrix

        for i in range(0,len(ind)):

            S_num[int(indices[i,0]), int(indices[i,1])] = ind[i] + 1

        # Solve sudoku matrix
        print(S_num)
        sud = np.zeros((9,9,10))
        sud[:,:,0] = S_num[:,:]
        # Cound occurences
        for k in range(0, 9):
            for l in range(0, 9):
                if sud[k,l,0] == 0:
                    for j in range (1,10):
                        if (((np.sum(sud[k,:,0] == j) == 0) & (np.sum(sud[:,l,0] == j) == 0)) & \
                        (np.sum(np.reshape(sud[(k / 3) * 3: (k / 3 + 1) * 3, (l / 3) * 3 : (l / 3 + 1) * 3, 0], (9, )) == j) == 0)):
                            sud[k,l,j] = j


        print(sud[:,:,1])

        tmp_sud = np.zeros((9, 9))
        #print(np.sum(np.reshape(tmp_sud - sud[:,:,0], (81,)) == 0))
        while np.sum(np.reshape(tmp_sud - sud[:,:,0], (81,)) == 0) != 81:

            tmp_sud = sud[:,:,0]

            for j in range (1,10):
                for k in range(0, 9):
                    for l in range(0, 9):
                        if sud[k,l,j] == j:
                            if (np.sum(sud[k,:,j] == j) == 1) | (np.sum(sud[:,l,j] == j) == 1) | \
                            (np.sum(np.reshape(sud[(k / 3) * 3: (k / 3 + 1) * 3, (l / 3) * 3 : (l / 3 + 1) * 3, j], (9, )) == j) == 1):
                                sud[k,l,0] = j
                                sud[k,l,1:10] = 0
                                sud[:,l,j] = 0
                                sud[k,:,j] = 0
                                sud[(k / 3) * 3: (k / 3 + 1) * 3, (l / 3) * 3 : (l / 3 + 1) * 3, j] = 0


        # Find squares with only two possibilities

        saved_ind  = np.zeros((5,2))
        saved_poss = np.zeros((5,10))
        prev_sud = np.zeros((9,9,10))
        for k in range (0,9):
            for l in range(0, 9):
                if sud[k,l,0] == 0:
                    if (np.sum(sud[k,l,:] != 0) == 2):
                        prev_sud[:,:,:] = sud[:,:,:]
                        saved_ind[0,:] = [k, l]
                        saved_poss[0,:] = sud[k,l,:]
                        guessvec = np.nonzero(sud[k,l,:])
                        guess = guessvec[0][random.randint(0,1)]
                        sud[k,l,0] = guess
                        sud[k,l,1:10] = 0
                        sud[:,l,guess] = 0
                        sud[k,:,guess] = 0
                        sud[(k / 3) * 3: (k / 3 + 1) * 3, (l / 3) * 3 : (l / 3 + 1) * 3, guess] = 0

                        tmp_sud = np.zeros((9, 9))

                        #print(np.sum(np.reshape(tmp_sud - sud[:,:,0], (81,)) == 0))
                        maxiter = 0
                        while maxiter < 100: #np.sum(np.reshape(tmp_sud - sud[:,:,0], (81,)) == 0) != 81:
                            maxiter = maxiter + 1
                            tmp_sud = sud[:,:,0]

                            for j in range (1,10):
                                for k in range(0, 9):
                                    for l in range(0, 9):
                                        if sud[k,l,j] == j:
                                            if (np.sum(sud[k,:,j] == j) == 1) | (np.sum(sud[:,l,j] == j) == 1) | \
                                            (np.sum(np.reshape(sud[(k / 3) * 3: (k / 3 + 1) * 3, (l / 3) * 3 : (l / 3 + 1) * 3, j], (9, )) == j) == 1):
                                                sud[k,l,0] = j
                                                sud[k,l,1:10] = 0
                                                sud[:,l,j] = 0
                                                sud[k,:,j] = 0
                                                sud[(k / 3) * 3: (k / 3 + 1) * 3, (l / 3) * 3 : (l / 3 + 1) * 3, j] = 0

                        if np.sum(np.reshape(sud[:,:,0], (81,)) == 0) != 0:
                            sud[:,:,:] = prev_sud[:,:,:]
                        else:
                            break

        # Draw number on picture

        print(sud[:,:,0])
        print(sud[:,:,1])
        print(sud[:,:,2])
        print(sud[:,:,3])
        print(sud[:,:,4])
        print(sud[:,:,5])
        print(sud[:,:,6])
        print(sud[:,:,7])
        print(sud[:,:,8])
        print(sud[:,:,9])

        S_num = sud[:,:,0] #np.random.randint(9, size=(9, 9))

        for k in range(0, len(empt_indices)):
            [row, col] = empt_indices[k,:]
            number = S_num[row, col]
            if number != 0:
                f[M[row,0] + n_mean[row,col,0] - def_size/2.0:M[row,0] + n_mean[row,col,0] + def_size/2.0, M[col,2] + n_mean[row,col,1] - def_size/2.0:M[col,2] + n_mean[row,col,1] + def_size/2.0, 0] = best_num_actual[:,:,number-1]
                f[M[row,0] + n_mean[row,col,0] - def_size/2.0:M[row,0] + n_mean[row,col,0] + def_size/2.0, M[col,2] + n_mean[row,col,1] - def_size/2.0:M[col,2] + n_mean[row,col,1] + def_size/2.0, 1] = best_num_actual[:,:,number-1]
                f[M[row,0] + n_mean[row,col,0] - def_size/2.0:M[row,0] + n_mean[row,col,0] + def_size/2.0, M[col,2] + n_mean[row,col,1] - def_size/2.0:M[col,2] + n_mean[row,col,1] + def_size/2.0, 2] = best_num_actual[:,:,number-1]



        print('S_num = ', S_num)
        #crop_face = f[lx / 4: - lx / 4, ly / 4: - ly / 4, lz / 4: - lz / 4]
        #flip_ud_face = np.flipud(f)
        #rotate_face = ndimage.rotate(f, 45)
        #rotate_face_noreshape = ndimage.rotate(f, 45, reshape=False)
        '''
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\One.png',   f[M[0,0] + mean_num[0,0] - 15:M[0,0] + mean_num[0,0] + 15, M[0,2] + mean_num[0,1] - 15:M[0,2] + mean_num[0,1] + 15, 2])
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\Two.png',   f[M[1,0] + mean_num[1,0] - 15:M[1,0] + mean_num[1,0] + 15, M[4,2] + mean_num[1,1] - 15:M[4,2] + mean_num[1,1] + 15, 2])
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\Three.png', f[M[2,0] + mean_num[2,0] - 15:M[2,0] + mean_num[2,0] + 15, M[8,2] + mean_num[2,1] - 15:M[8,2] + mean_num[2,1] + 15, 2])
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\Four.png',  f[M[2,0] + mean_num[3,0] - 15:M[2,0] + mean_num[3,0] + 15, M[3,2] + mean_num[3,1] - 15:M[3,2] + mean_num[3,1] + 15, 2])
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\Five.png',  f[M[2,0] + mean_num[4,0] - 15:M[2,0] + mean_num[4,0] + 15, M[4,2] + mean_num[4,1] - 15:M[4,2] + mean_num[4,1] + 15, 2])
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\Six.png',   f[M[2,0] + mean_num[5,0] - 15:M[2,0] + mean_num[5,0] + 15, M[5,2] + mean_num[5,1] - 15:M[5,2] + mean_num[5,1] + 15, 2])
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\Seven.png', f[M[3,0] + mean_num[6,0] - 15:M[3,0] + mean_num[6,0] + 15, M[5,2] + mean_num[6,1] - 15:M[5,2] + mean_num[6,1] + 15, 2])
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\Eight.png', f[M[2,0] + mean_num[7,0] - 15:M[2,0] + mean_num[7,0] + 15, M[2,2] + mean_num[7,1] - 15:M[2,2] + mean_num[7,1] + 15, 2])
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\\Nine.png', f[M[2,0] + mean_num[8,0] - 15:M[2,0] + mean_num[8,0] + 15, M[1,2] + mean_num[8,1] - 15:M[1,2] + mean_num[8,1] + 15, 2])
        '''
        misc.imsave('C:\Users\Markus\Documents\Python Scripts\Image processing\graceNew.png', f)
        plt.imshow(f)
        plt.show()
        '''
        plt.imshow(f[M[3,0]:M[3,0] + 45, M[3,2]:M[3,2] + 45, 2] - f[M[2,0]:M[2,0] + 45, M[2,2]:M[2,2] + 45, 2])
        plt.show()

        plt.imshow(np.reshape(RealNumbers[:,2], (45, 45)))
        plt.show()
        plt.imshow(np.reshape(RealNumbers[:,3], (45, 45)))
        plt.show()
        plt.imshow(np.reshape(RealNumbers[:,4], (45, 45)))
        plt.show()
        plt.imshow(np.reshape(RealNumbers[:,5], (45, 45)))
        plt.show()
        plt.imshow(np.reshape(RealNumbers[:,6], (45, 45)))
        plt.show()
        plt.imshow(np.reshape(RealNumbers[:,7], (45, 45)))
        plt.show()
        plt.imshow(np.reshape(RealNumbers[:,8], (45, 45)))
        plt.show()
        '''
Im = image_class('face.png')
