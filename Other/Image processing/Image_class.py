class image_class:
    def __init__(self, source):
        import numpy as np
        from scipy import misc
        from scipy import ndimage
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import math
        import random

        #img = mpimg.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/stinkbug.png')
        self.f = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/Sudoku00.png')

        # 255 = White
        # 0 = Black

        self.lx, self.ly, self.lz = self.f.shape

        print(self.lx, self.ly, self.lz)

        self.f[:, :, 0] = (self.f[:, :, 0] > 220) * 255
        self.f[:, :, 1] = (self.f[:, :, 1] > 220) * 255
        self.f[:, :, 2] = (self.f[:, :, 2] > 220) * 255

        plt.imshow(self.f)
        plt.show()

    def run(self):

        self.findLine()
        self.findSquares()
        self.findOcupiedSquares()
        self.CutOutNumbers()
        self.ClassifyNumbers()

        self.SolveSudokuMatrix()


    def findLine(self):
        import numpy as np
        import math

        L  = np.zeros((self.ly, 1))
        L2 = np.zeros((self.ly, 1))
        R  = np.zeros((self.lx, 1))
        R2 = np.zeros((self.lx, 1))

        for k in range(0, self.ly):
            L[k] = (np.sum(self.f[:, k, 0]) + np.sum(self.f[:, k, 1]) + np.sum(self.f[:, k, 2]))/len(L2)
            L2[k] = math.pow(np.sum(self.f[:, k, 0]) + np.sum(self.f[:, k, 1]) + np.sum(self.f[:, k, 2]), 2)

        for l in range(0, self.lx):
            R[l] = (np.sum(self.f[l, :, 0]) + np.sum(self.f[l, :, 1]) + np.sum(self.f[l, :, 2]))/len(R2)
            R2[l] = math.pow(np.sum(self.f[l, :, 0]) + np.sum(self.f[l, :, 1]) + np.sum(self.f[l, :, 2]), 2)

        VarL = L2/len(L2) - np.square(L)/len(L)
        VarR = R2/len(R2) - np.square(R)/len(R)
        stdL = np.sqrt(VarL)
        stdR = np.sqrt(VarR)

        for k in range(0, self.ly):
            if stdL[k] < 1000:
                self.f[:, k, 0] = 0
                self.f[:, k, 1] = 255
                self.f[:, k, 2] = 0

        for l in range(0, self.lx):
            if stdR[l] < 1000:
                self.f[l, :, 0] = 0
                self.f[l, :, 1] = 255
                self.f[l, :, 2] = 0

    def findSquares(self):
        import numpy as np
        import matplotlib.pyplot as plt

        squares = 9
        self.M = np.zeros((squares, 4))
        tmprow = 0
        tmpcol = 0
        tmprowEnd = 0
        tmpcolEnd = 0

        for k in range(1, self.ly):

            if (((self.f[round(self.lx/(squares*2)), self.ly - k, 0] == 0) & \
                (self.f[round(self.lx/(squares*2)), self.ly - k, 1] == 255) & \
                (self.f[round(self.lx/(squares*2)), self.ly - k, 2] == 0)) & \
                ((self.f[round(self.lx/(squares*2)), self.ly - k - 1, 0] == 255) & \
                (self.f[round(self.lx/(squares*2)), self.ly - k - 1, 1] == 255) & \
                (self.f[round(self.lx/(squares*2)), self.ly - k - 1, 2] == 255))) & (tmpcolEnd < squares):
                self.M[squares - 1 - tmpcolEnd, 3] = self.ly - k
                tmpcolEnd = tmpcolEnd + 1

            if ((self.f[round(self.lx/(squares*2)), k-1, 0] == 0) & \
                (self.f[round(self.lx/(squares*2)), k-1, 1] == 255) & \
                (self.f[round(self.lx/(squares*2)), k-1, 2] == 0)) & \
                ((self.f[round(self.lx/(squares*2)), k, 0] == 255) & \
                (self.f[round(self.lx/(squares*2)), k, 1] == 255) & \
                (self.f[round(self.lx/(squares*2)), k, 2] == 255)) & (tmpcol < squares):

                self.M[tmpcol, 2] = k
                tmpcol = tmpcol + 1

        for l in range(1, self.lx):

            if (((self.f[self.lx - l, round(self.ly/(squares*2)), 0] == 0) & \
                (self.f[self.lx - l, round(self.ly/(squares*2)), 1] == 255) & \
                (self.f[self.lx - l, round(self.ly/(squares*2)), 2] == 0)) & \
                ((self.f[self.lx - l - 1, round(self.ly/(squares*2)), 0] == 255) & \
                (self.f[self.lx - l - 1, round(self.ly/(squares*2)), 1] == 255) & \
                (self.f[self.lx - l - 1, round(self.ly/(squares*2)), 2] == 255)) & (tmprowEnd < squares)):

                self.M[squares - tmprowEnd - 1, 1] = self.lx - l
                tmprowEnd = tmprowEnd + 1

            if (((self.f[l-1, round(self.ly/(squares*2)), 0] == 0) & \
                (self.f[l-1, round(self.ly/(squares*2)), 1] == 255) & \
                (self.f[l-1, round(self.ly/(squares*2)), 2] == 0)) & \
                ((self.f[l, round(self.ly/(squares*2)), 0] == 255) & \
                (self.f[l, round(self.ly/(squares*2)), 1] == 255) & \
                (self.f[l, round(self.ly/(squares*2)), 2] == 255)) & (tmprow < squares)):

                self.M[tmprow, 0] = l
                tmprow = tmprow + 1

        plt.imshow(self.f)
        plt.show()

    def findOcupiedSquares(self):
        import numpy as np
        # Find which squares including numbers

        self.S_num = np.zeros((9, 9))
        self.default_num = 36 #27 #30 #35
        self.def_size = 30
        self.Numbers = np.zeros((self.def_size, self.def_size, self.default_num))
        tmp = 0
        tmp2 = 0
        self.indices = np.zeros((81, 2))
        self.empt_indices = np.zeros((81, 2))
        self.n_min = np.zeros((9,9,2))
        self.n_max = np.zeros((9,9,2))
        self.n_mean = np.zeros((9,9,2))

        print(self.M)
        for k in range(0, 9):

            for l in range(0, 9):

                if (np.mean(self.f[int(self.M[l,0]):int(self.M[l,1]), int(self.M[k,2]):int(self.M[k,3]), 0]) < 250):
                       self.S_num[l, k] = 10
                       self.indices[tmp, :] = [l, k]
                       self.n_min[l, k, :]  = np.min(np.where(self.f[int(self.M[l,0]):int(self.M[l,1]), int(self.M[k,2]):int(self.M[k,3]), 0] < 10), 1)
                       self.n_max[l, k, :]  = np.max(np.where(self.f[int(self.M[l,0]):int(self.M[l,1]), int(self.M[k,2]):int(self.M[k,3]), 0] < 10), 1)
                       self.n_mean[l, k, :] = np.round((self.n_max[l,k,:] + self.n_min[l,k,:])/2.0)

                       self.Numbers[:,:,tmp] = self.f[int(self.M[l,0] + self.n_mean[l,k,0] - self.def_size/2.0):int(self.M[l,0] + self.n_mean[l,k,0] + self.def_size/2.0), int(self.M[k,2] + self.n_mean[l,k,1] - self.def_size/2.0):int(self.M[k,2] + self.n_mean[l,k,1] + self.def_size/2.0), 1]

                       if (self.n_mean[l, k, 0] - self.def_size/2.0 < 0):
                           shiftrow = 1
                       else:
                           shiftrow = 0
                       if (self.n_mean[l, k, 1] - self.def_size/2.0 < 0):
                           shiftcol = 1
                       else:
                           shiftcol = 0

                       self.n_min[l, k, :] = np.min(np.where(self.f[int(self.M[l,0] + self.n_mean[l,k,0] - self.def_size/2.0 + shiftrow):int(self.M[l,0] + self.n_mean[l,k,0] + self.def_size/2.0), int(self.M[k,2] + self.n_mean[l,k,1] - self.def_size/2.0 + shiftcol):int(self.M[k,2] + self.n_mean[l,k,1] + self.def_size/2.0), 2] < 100), 1)
                       self.n_max[l, k, :] = np.max(np.where(self.f[int(self.M[l,0] + self.n_mean[l,k,0] - self.def_size/2.0 + shiftrow):int(self.M[l,0] + self.n_mean[l,k,0] + self.def_size/2.0), int(self.M[k,2] + self.n_mean[l,k,1] - self.def_size/2.0 + shiftcol):int(self.M[k,2] + self.n_mean[l,k,1] + self.def_size/2.0), 2] < 100), 1)

                       tmp = tmp + 1

                else:
                   self.n_min[l, k, :]  = np.min(np.where(self.f[int(self.M[l,0]):int(self.M[l,1]), int(self.M[k,2]) :int(self.M[k,3]), 0] > 100), 1)
                   self.n_max[l, k, :]  = np.max(np.where(self.f[int(self.M[l,0]):int(self.M[l,1]), int(self.M[k,2]) :int(self.M[k,3]), 0] > 100), 1)
                   self.n_mean[l, k, :] = np.round((self.n_max[l, k, :] + self.n_min[l, k, :])/2.0)
                   self.empt_indices[tmp2, :] = [l, k]
                   tmp2 = tmp2 + 1

    def CutOutNumbers(self):
        import numpy as np
        from scipy import misc
        import math
        # Cut out numbers

        RealNumbers = np.zeros((30,30,9))

        min_num = np.zeros((9,2))
        max_num = np.zeros((9,2))
        mean_num = np.zeros((9,2))
        min_num[:, 0] = [5,6,6,6,6,6,6,6,6]
        min_num[:, 1] = [11,8,8,8,8,8,8,8,8]
        max_num[:, 0] = [23,25,25,25,25,25,25,25,25]
        max_num[:, 1] = [20,22,22,22,22,22,22,22,22]
        mean_num[:, 0] = [24,24,24,24,24,24,24,24,24]
        mean_num[:, 1] = [25,26,26,27,27,27,26,26,26]

        RealNumbers[:, :, 0] = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/One.png')
        RealNumbers[:, :, 1] = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/Two.png')
        RealNumbers[:, :, 2] = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/Three.png')
        RealNumbers[:, :, 3] = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/Four.png')
        RealNumbers[:, :, 4] = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/Five.png')
        RealNumbers[:, :, 5] = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/Six.png')
        RealNumbers[:, :, 6] = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/Seven.png')
        RealNumbers[:, :, 7] = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/Nine.png')
        RealNumbers[:, :, 8] = misc.imread('C:/Users/Markus/Documents/Python Scripts/Other/Image processing/Eight.png')

        # Interpolation
        self.RealNumbers_scaled = np.zeros((self.def_size, self.def_size, 9))

        for k in range(0, 9):

            dx                                               = float(self.def_size)/(max_num[k, 0] - min_num[k, 0] + 1)
            self.RealNumbers_scaled[0, :, k]                 = RealNumbers[int(min_num[k, 0]), :, k]
            self.RealNumbers_scaled[self.def_size - 1, :, k] = RealNumbers[int(max_num[k, 0]), :, k]

            for l in range(1, self.def_size - 1):

                div                              = math.floor(l/dx)
                rest                             = l/dx - math.floor(l/dx)
                self.RealNumbers_scaled[l, :, k] = (1 - rest) * RealNumbers[int(min_num[k, 0] + div), :, k] + (rest) * RealNumbers[int(min_num[k, 0] + div + 1), :, k]

        tmpScaled          = np.zeros((self.def_size, self.def_size, 9))
        tmpScaled[:, :, :] = self.RealNumbers_scaled[:, :, :]

        for k in range(0, 9):

            dx                                               = float(self.def_size)/(max_num[k, 1] - min_num[k, 1] + 1)
            self.RealNumbers_scaled[:, 0, k]                 = tmpScaled[:, int(min_num[k, 1]), k]
            self.RealNumbers_scaled[:, self.def_size - 1, k] = tmpScaled[:, int(max_num[k, 1]), k]

            for l in range(1, self.def_size - 1):

                div                         = math.floor(l/dx)
                rest                        = l/dx - math.floor(l/dx)
                self.RealNumbers_scaled[:, l, k] = (1 - rest) * tmpScaled[:, int(min_num[k, 1] + div), k] + (rest)*tmpScaled[:, int(min_num[k, 1] + div + 1), k]

        # Interpolate Numbers
        self.Numbers_scaled = np.zeros((30, 30, self.default_num))

        for k in range(0, self.default_num):
            [l1, k1]                      = self.indices[k, :]
            dx                            = 30.0/(self.n_max[int(l1), int(k1), 0] - self.n_min[int(l1), int(k1), 0] + 1)
            self.Numbers_scaled[0, :, k]  = self.Numbers[int(self.n_min[int(l1), int(k1), 0]), :, k]
            self.Numbers_scaled[29, :, k] = self.Numbers[int(self.n_max[int(l1), int(k1), 0]), :, k]

            for l in range(1,29):

                div                          = math.floor(l/dx)
                rest                         = l/dx - math.floor(l/dx)
                self.Numbers_scaled[l, :, k] = (1 - rest) * self.Numbers[int(self.n_min[int(l1), int(k1), 0] + div), :, k] + (rest) * self.Numbers[int(self.n_min[int(l1), int(k1), 0] + div + 1), :, k]

        tmpScaled          = np.zeros((30, 30, self.default_num))
        tmpScaled[:, :, :] = self.Numbers_scaled[:, :, :]

        for k in range(0, self.default_num):
            [l1, k1]                      = self.indices[k, :]
            dx                            = 30.0/(self.n_max[int(l1), int(k1), 1] - self.n_min[int(l1), int(k1), 1] + 1)
            self.Numbers_scaled[:, 0, k]  = tmpScaled[:, int(self.n_min[int(l1), int(k1), 1]), k]
            self.Numbers_scaled[:, 29, k] = tmpScaled[:, int(self.n_max[int(l1), int(k1), 1]), k]

            for l in range(1, 29):
                div                          = math.floor(l/dx)
                rest                         = l/dx - math.floor(l/dx)
                self.Numbers_scaled[:, l, k] = (1 - rest) * tmpScaled[:, int(self.n_min[int(l1), int(k1), 1] + div), k] + (rest) * tmpScaled[:, int(self.n_min[int(l1), int(k1), 1] + div + 1), k]

    def ClassifyNumbers(self):
        import numpy as np
        # Classify numbers

        dist                 = np.zeros((9, 1))
        ind                  = np.zeros((self.default_num, 1))
        min_dist             = 10*255*255*30*30*np.ones((9, 1))
        self.best_numbers    = np.zeros((30, 30, 9))
        self.best_num_actual = np.zeros((30, 30, 9))

        for k in range(0, self.default_num): #len(ind)

            dist = np.zeros((9, 1))

            for l in range(0, 9):

                dist[l] = np.sum(np.square(np.reshape(self.Numbers_scaled[:, :, k], (30*30,)) - np.reshape(self.RealNumbers_scaled[:, :, l], (30*30,))))

                if dist[l] < min_dist[l]:
                    min_dist[l]            = dist[l]
                    self.best_numbers[:,:,l]    = self.Numbers_scaled[:,:,k]
                    self.best_num_actual[:,:,l] = self.Numbers[:,:,k]

        # Run again with best numbers
        for k in range(0, self.default_num):

            dist = np.zeros((9,1))

            for l in range(0,9):

                dist[l] = np.sum(np.square(np.reshape(self.Numbers_scaled[:,:,k], (30*30,)) - np.reshape(self.best_numbers[:,:,l], (30*30,))))

            ind[k] = np.argmin(dist)

        # Create sudoku matrix
        for i in range(0, len(ind)):

            self.S_num[int(self.indices[i,0]), int(self.indices[i,1])] = ind[i] + 1

    def SolveSudokuMatrix(self):
        import numpy as np
        import random
        from scipy import misc
        import matplotlib.pyplot as plt
        # Solve sudoku matrix

        sud = np.zeros((9, 9, 10))
        sud[:,:,0] = self.S_num[:, :]
        # Cound occurences
        for k in range(0, 9):
            for l in range(0, 9):
                if sud[k,l,0] == 0:
                    for j in range (1, 10):

                        if (((np.sum(sud[k, :, 0] == j) == 0) & (np.sum(sud[:, l, 0] == j) == 0)) & \
                        (np.sum(np.reshape(sud[int(k / 3) * 3: (int(k / 3) + 1) * 3, int(l / 3) * 3 : (int(l / 3) + 1) * 3, 0], (9, )) == j) == 0)):
                            sud[k, l, j] = j

        tmp_sud = np.zeros((9, 9))

        while np.sum(np.reshape(tmp_sud - sud[:,:,0], (81,)) == 0) != 81:

            tmp_sud = sud[:, :, 0]

            for j in range (1, 10):
                for k in range(0, 9):
                    for l in range(0, 9):
                        if sud[k, l, j] == j:
                            if (np.sum(sud[k, :, j] == j) == 1) | (np.sum(sud[:, l, j] == j) == 1) | \
                            (np.sum(np.reshape(sud[int(k / 3) * 3: (int(k / 3) + 1) * 3, int(l / 3) * 3 : (int(l / 3) + 1) * 3, j], (9, )) == j) == 1):
                                sud[k,l,0] = j
                                sud[k,l,1:10] = 0
                                sud[:,l,j] = 0
                                sud[k,:,j] = 0
                                sud[int(k / 3) * 3: (int(k / 3) + 1) * 3, int(l / 3) * 3 : (int(l / 3) + 1) * 3, j] = 0

        # Find squares with only two possibilities
        saved_ind  = np.zeros((5, 2))
        saved_poss = np.zeros((5, 10))
        prev_sud   = np.zeros((9, 9, 10))

        for k in range (0, 9):
            for l in range(0, 9):
                if sud[k,l,0] == 0:
                    if (np.sum(sud[k, l, :] != 0) == 2):
                        prev_sud[:, :, :] = sud[:, :, :]
                        saved_ind[0, :]   = [k, l]
                        saved_poss[0, :]  = sud[k, l, :]
                        guessvec          = np.nonzero(sud[k, l, :])
                        guess             = guessvec[0][random.randint(0, 1)]
                        sud[k, l, 0]      = guess
                        sud[k, l, 1:10]   = 0
                        sud[:, l, guess]  = 0
                        sud[k, :, guess]  = 0
                        sud[int(k / 3) * 3: (int(k / 3) + 1) * 3, int(l / 3) * 3 : (int(l / 3) + 1) * 3, guess] = 0

                        tmp_sud = np.zeros((9, 9))
                        maxiter = 0

                        while maxiter < 100:
                            maxiter = maxiter + 1
                            tmp_sud = sud[:,:,0]

                            for j in range (1,10):
                                for k in range(0, 9):
                                    for l in range(0, 9):
                                        if sud[k,l,j] == j:
                                            if (np.sum(sud[k,:,j] == j) == 1) | (np.sum(sud[:,l,j] == j) == 1) | \
                                            (np.sum(np.reshape(sud[int(k / 3) * 3: (int(k / 3) + 1) * 3, int(l / 3) * 3 : (int(l / 3) + 1) * 3, j], (9, )) == j) == 1):
                                                sud[k, l, 0] = j
                                                sud[k, l, 1:10] = 0
                                                sud[:, l, j] = 0
                                                sud[k, :, j] = 0
                                                sud[int(k / 3) * 3: (int(k / 3) + 1) * 3, int(l / 3) * 3 : (int(l / 3) + 1) * 3, j] = 0

                        if np.sum(np.reshape(sud[:, :, 0], (81,)) == 0) != 0:
                            sud[:, :, :] = prev_sud[:, :, :]
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

        self.S_num = sud[:, :, 0]

        for k in range(0, len(self.empt_indices)):
            [row, col] = self.empt_indices[k,:]
            row = int(row)
            col = int(row)
            number = int(self.S_num[int(row), int(col)])
            if number != 0:

                self.f[int(self.M[row,0] + self.n_mean[row,col,0] - self.def_size/2.0):int(self.M[row,0] + self.n_mean[row,col,0] + self.def_size/2.0), int(self.M[col,2] + self.n_mean[row,col,1] - self.def_size/2.0):int(self.M[col,2] + self.n_mean[row,col, 1] + self.def_size/2.0), 0] = self.best_num_actual[:, :, number-1]
                self.f[int(self.M[row,0] + self.n_mean[row,col,0] - self.def_size/2.0):int(self.M[row,0] + self.n_mean[row,col,0] + self.def_size/2.0), int(self.M[col,2] + self.n_mean[row,col,1] - self.def_size/2.0):int(self.M[col,2] + self.n_mean[row,col, 1] + self.def_size/2.0), 1] = self.best_num_actual[:, :, number-1]
                self.f[int(self.M[row,0] + self.n_mean[row,col,0] - self.def_size/2.0):int(self.M[row,0] + self.n_mean[row,col,0] + self.def_size/2.0), int(self.M[col,2] + self.n_mean[row,col,1] - self.def_size/2.0):int(self.M[col,2] + self.n_mean[row,col, 1] + self.def_size/2.0), 2] = self.best_num_actual[:, :, number-1]

        print('S_num = ', self.S_num)

        misc.imsave('graceNew.png', self.f)
        plt.imshow(self.f)
        plt.show()

Im = image_class('face.png')
Im.run()
