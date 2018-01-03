class Roulette:
    def __init__(self, a):
        import numpy as np
        self.N = 1000
        self.M = 1000
        self.bet = 5
        self.startBalance = 500

    def RouletteSpin(self):
        import numpy as np

        for i in range(0, self.M - 1):
            pot = self.bet
            balance = self.startBalance - pot
            account[i][0] = balance + pot
            randnum = np.random.randint(2, size=self.N)

            for j in range(1, self.N - 1):

                if randnum[j]:
                    balance = balance + pot/2
                    pot = 1.5*pot
                else:
                    balance = balance - self.bet
                    pot = 0

                account[i][j] = balance + pot

        return account

    def plotSolutions(self):
        return t**2

    def boundaryData_West(t):
        return t**2
