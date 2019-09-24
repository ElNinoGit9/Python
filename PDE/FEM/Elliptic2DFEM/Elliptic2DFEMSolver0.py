class gridClass:
    'Common base class for all employees'
    def __init__(self, minv, maxv, N):
        import numpy as np
        self.minx = minv[0]
        self.maxx = maxv[0]
        self.miny = minv[1]
        self.maxy = maxv[1]
        self.Lx   = N[0]
        self.Ly   = N[1]
        self.Nt   = N[2]

        self.generateMesh()

    def generateMesh(self):
        import numpy as np

        x = np.linspace(self.minx, self.maxx, self.Lx, endpoint=True)
        y = np.linspace(self.miny, self.maxy, self.Ly, endpoint=True)

        gridY, gridX = np.meshgrid(y, x)

        gridX = gridX.flatten()
        gridY = gridY.flatten()

        self.p = np.matrix([gridY, gridX])

        self.t = np.zeros((2*(self.Lx-1)*(self.Ly-1),4))
        k = 0
        for r in range(0,self.Ly-1):
            for c in range(0, self.Lx-1):

                self.t[k,   :] = [c + r*self.Lx,   c+1 + r*self.Lx,   (r+1)*self.Lx + c, 1]
                self.t[k+1, :] = [c+1 + r*self.Lx, (r+1)*self.Lx + c, (r+1)*self.Lx + c + 1, 1]
                k = k + 2

        self.t = np.transpose(self.t)
        A = range(0, self.Lx)
        A.extend(range(2*(self.Lx - 1)+1, self.Lx*self.Ly, self.Ly))
        A.extend(range(self.Lx * self.Ly-2, self.Lx * (self.Ly-1)-1, -1))
        A.extend(range(self.Lx*(self.Ly-2), 0, -self.Ly))

        B = A[1:]
        B.append(A[0])

        C = np.squeeze(np.asarray(np.zeros((1,len(B)))))
        D = np.squeeze(np.asarray(np.ones((1,len(B)))))
        E = range(0,len(B))
        F = np.squeeze(np.asarray(np.ones((1,len(B)))))
        G = np.squeeze(np.asarray(np.zeros((1,len(B)))))

        self.e = np.matrix([A,B,C,D,E,F,G])

    def plot_grid(self):
        import matplotlib.pyplot as plt
        import numpy as np
        print "plotting grid"

        numTri = len(np.squeeze(np.asarray(self.t[0, :])))
        X = np.zeros((4, numTri))
        Y = np.zeros((4, numTri))

        for k in range(0, numTri):

            X[0,k] = self.p[0, self.t[0, k]]
            X[1,k] = self.p[0, self.t[1, k]]
            X[2,k] = self.p[0, self.t[2, k]]
            X[3,k] = X[0,k]
            Y[0,k] = self.p[1, self.t[0, k]]
            Y[1,k] = self.p[1, self.t[1, k]]
            Y[2,k] = self.p[1, self.t[2, k]]
            Y[3,k] = Y[0,k]
            plt.plot(X[:,k], Y[:,k])

        xlen = (self.maxx - self.minx) * 0.1
        ylen = (self.maxy - self.miny) * 0.1
        plt.axis([self.minx - xlen, self.maxx + xlen, self.miny - ylen, self.maxy + ylen])
        plt.show()

class problemClass:
    def __init__(self, grid):
        import numpy as np

        def a_func(x): return (3 + 0*x)
        def b_func(x): return (1 + 0*x)

        self.a = a_func
        self.b = b_func

    def initialData(self, data):

        def u_a(x, t):   return        sin(2*pi*(x - t))
        def uT_a(x, t):  return -2*pi* cos(2*pi*(x - t))
        def uX_a(x, t):  return  2*pi* cos(2*pi*(x - t))

    def createData(self, grid_obj):

        def gE(x, t): return self.u_a (x, t)
        def gW(x, t): return self.u_a (x, t)
        def f(x, t):  return self.u_a(x, t)

class SchemeClass:
    def __init__(self, problem):
        print 'Initialize'

    def Scheme(self, grid, problem):
        print 'Create scheme'
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import numpy as np

        def u   (x,y): return (16*x*y*(1-x)*(1-y));
        def uxx (x,y): return -32*(y*(1-y));
        def uyy (x,y): return -32*(x*(1-x));
        def f   (x,y): return (-uxx(x,y) - uyy(x,y));

        N = len(np.squeeze(np.asarray(grid.p[0, :])))

        [A, F] = self.computeStiffnessMatrix(grid)

        boundary = np.unique([grid.e[0,:], grid.e[1,:]])

        interior = [x for x in range(1, N) if x not in boundary]

        U = np.zeros((grid.Lx+1)*(grid.Ly+1))

        for k in range(0,len(boundary)):
            U[boundary[k]] = u(grid.t[0, boundary[k]] , grid.t[1,boundary[k]])

        for l in range(0,len(interior)):
            for m in range(0,len(boundary)):
                F[interior[l]] = F[interior[l]] - A[interior[l], boundary[m]] * U[boundary[m]]

        print F
        print U
        print A
        U[interior] = np.linalg.solve(A[[[val] for val in interior], [[interior]]], F[interior])

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot_trisurf(np.squeeze(np.asarray(grid.p[0,:])), np.squeeze(np.asarray(grid.p[1,:])), np.squeeze(np.asarray(u_h.T)), linewidth=0.2, antialiased=True)
        plt.show()

    def boundaryConditionImposition(self):
        print 'Impose boundary conditions'

    def computeStiffnessMatrix(self, grid):
        print 'Compute stiffness matrix'
        import numpy as np

        def u   (x,y): return (16*x*y*(1-x)*(1-y));
        def uxx (x,y): return -32*(y*(1-y));
        def uyy (x,y): return -32*(x*(1-x));
        def f   (x,y): return (-uxx(x,y) - uyy(x,y));

        A = np.zeros(((grid.Lx+1)*(grid.Ly+1), (grid.Lx+1)*(grid.Ly+1)))
        F = np.zeros((grid.Lx+1)*(grid.Ly+1))
        xk = np.zeros(3)
        yk = np.zeros(3)
        NT = len(grid.t)
        print NT
        for k in range(0, NT):

            xk[0] = grid.p[0, grid.t[0,k]]
            xk[1] = grid.p[0, grid.t[1,k]]
            xk[2] = grid.p[0, grid.t[2,k]]
            yk[0] = grid.p[1, grid.t[0,k]]
            yk[1] = grid.p[1, grid.t[1,k]]
            yk[2] = grid.p[1, grid.t[2,k]]
            Jk = [[xk[1]-xk[0], xk[2]-xk[0]], [yk[1]-yk[0], yk[2]-yk[0]]]
            Ak = np.abs(np.linalg.det(Jk))/2.
            Mk = [np.ones(3), xk, yk]
            Mk = np.matrix(Mk)

            Ck = np.linalg.inv(Mk)
            E2 = np.zeros((3,3))

            for i in range(0,2):
                for j in range(0,2):

                    E2[i,j] = Ak*np.transpose(Ck[2:3, i]) * Ck[2:3, j];

            tp = np.squeeze(np.asarray(grid.t[0:3, k], dtype = int))

            A[[[val] for val in tp], [[tp]]] = A[[[val] for val in tp], [[tp]]] + E2
            F[tp] = F[tp] + Ak*f(sum(xk)/3., sum(yk)/3.)/3.

        return A, F

    def basisLinear(self, x):
        print 'Compute linear basis'
        import numpy as np

        M = len(np.squeeze(np.asarray(x[0,:])))

        value = np.zeros((3, M))

        value[0, :] = np.ones((1,M)) - x[0, :] - x[1, :]
        value[1, :] = x[0, :]
        value[2, :] = x[1, :]

        d_value = np.zeros((2, M, 3))
        v = np.ones((1, M))

        d_value[:,:,0] = [-v, -v]
        d_value[:,:,1] = [v, np.zeros((1,M))]
        d_value[:,:,2] = [np.zeros((1,M)), v]

        return value, d_value

    def quadrature(self, order, elementType):
        import numpy as np

        if elementType is 'Triangle':
            return self.quadTriangle(order)
        elif elementType is 'Quadrilatural':
            return self.quadQuadlirateral(order)

    def quadTriangle(self, order):
        import numpy as np

        if order is 1:
            q = np.matrix([[1/3., 1/3.]])
            w = np.array([1/2.])
        elif order is 3:
            q = np.transpose(np.matrix([[0.5, 0], [0, 0.5], [0.5, 0.5]]))
            w = np.array([1/6., 1/6., 1/6.])
        elif order is 4:
            q = np.matrix([[1/3., 1/3.], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]])
            w = np.array([-27/96., 25/96., 25/96., 25/96])

        return q, w

    def quadQuadlirateral(self):
        import numpy as np

        if order is 1:
            q = np.matrix([[0, 0]])
            w = np.array([2])
        elif order is 4:
            q = np.matrix([[-np.sqrt(1/3.), -np.sqrt(1/3.)], [np.sqrt(1/3.), -np.sqrt(1/3.)], [-np.sqrt(1/3.), np.sqrt(1/3.)], [np.sqrt(1/3.), np.sqrt(1/3.)]])
            w = np.array([1, 1, 1, 1])
        elif order is 9:
            q = np.matrix([[-np.sqrt(3/5.), -np.sqrt(3/5.)], [0, -np.sqrt(3/5.)], [np.sqrt(3/5.), -np.sqrt(3/5.)], [-np.sqrt(3/5.), 0], [0, 0], \
            [np.sqrt(3/5.), 0], [-np.sqrt(3/5.), np.sqrt(3/5.)], [0, np.sqrt(3/5.)], [np.sqrt(3/5.), np.sqrt(3/5.)]])
            w = np.array([25/81., 40/81., 25/81., 40/81., 64/81., 40/81., 25/81., 40/81., 25/81.])

    def Error(self, grid, problem):
        print 'Error'
        import matplotlib.pyplot as plt
        import numpy as np

        N = 20
        xv = grid.grid
        xn = np.linspace(grid.minx, grid.maxx, N + 1, endpoint=True)
        self.un = np.squeeze(np.asarray(np.zeros((1, N+1))))

        for k in range(0, N):
            self.un[k] = self.fem_sol(xv, xn[k])


        def u_a(x):    return np.sin(np.pi*x)

        error = np.abs(u_a(xn) - self.un)
        self.error_L2 = np.sqrt(np.dot(error.T, error))

        print self.error_L2

        plt.plot(xn, u_a(xn), xn, self.un)
        plt.axis([0,1,0,np.max([np.max(u_a(xn)), np.max(self.un)])])
        plt.show()

        plt.plot(xn, error)
        plt.axis([grid.minx, grid.maxx,0,np.max(error)])
        plt.show()

g = gridClass([0, 0, 0], [1, 1, 1], [5, 5, 4000])
g.plot_grid()

p = problemClass(g)
p.createData(g)

s = SchemeClass(p)
s.Scheme(g, p)
s.Error(g, p)
