"""
The implementation of the polynomials in Theorem 3.1. 
The current state of the program is tested against the equality such that when m = 1, l.h.s. of iii equals r.h.s. of iii minus 1. Also, it is tested against the value of P^i_0(1) and P^i_1(1). To be more specific, the program has to produce, for \alpha = 0.75 and m = 6:
P^1_0 = 1; P^2_0 = 6; P^3_0 = 56; P^4_0 = 556; 
P^1_1 = 4; P^2_1 = 44; P^3_1 = 412; P^4_1 = 4124;
This program can be also a template for other polynomials that we can propose. Also, it may be useful when making plots in the paper. 
"""

import numpy as np

class error_poly:
    def __init__(self, alpha: float, m: int):
        self.alpha = alpha # \alpha in the paper
        self.m = m # m = 1 / \Delta t in the paper
        self.factor = 1.0 / (1.0 - alpha) # for convenience
        self.p10 = 1.0 # P^1_0(x)
        self.p11 = self.factor # P^1_1(x)


    def p0(self, i: int, x: np.ndarray) -> np.ndarray:
        """
        P^i_0(x)
        """
        if i == 1:
            return self.p10 * np.ones_like(x)
        elif i == 2:
            return 1.0 + (1.0 + self.factor) * x
        elif i == 3:
            return ((self.m == 1) + (1 + 2 * self.factor) * x) * self.p0(2, x) + (x + 1) * (self.m >= 2) 
        else:
            return (1 + (1 + 2 * self.factor) * x) * self.p0(i - 1, x) - self.factor * x * (i <= self.m + 1)
        

    def p1(self, i: int, x: np.ndarray) -> np.ndarray:
        """
        P^i_1(x)
        """
        if i == 1:
            return self.p11 * np.ones_like(x)
        elif i == 2:
            return self.factor * (1 + 2 * (1 + 1 * self.factor) * x + (self.m == 1))
        elif i == 3:
            return ((self.m == 1) + (1 + 2 * self.factor) * x) * self.p1(2, x) + self.factor * (3 * x + 1) * (self.m >= 2) + self.factor * (self.m == 2)
        else:
            return (1 + (1 + 2 * self.factor) * x) * self.p1(i - 1, x) + self.factor * x * (i <= self.m) + (self.m >= 3) * (self.m + 1 == i)
        

    def sum(self, id: int, i: int, x: np.ndarray) -> np.ndarray:
        """
        Sum of the first i terms, id = 0 or 1 indicates P_0 or P_1 respectively
        """
        if i < 0:
            print('Invalid sum')
        term = eval("self.p{}".format(id))
        result = np.zeros_like(x)
        for k in range(i):
            result += term(k + 1, x)
        return result


    def iii_lhs(self, i: int, x: np.ndarray) -> np.ndarray:
        """
        l.h.s. of condition (iii)
        """
        return self.p0(i, x) +  x * self.sum(0, i - 1, x) + x * self.sum(1, i - 1, x)


    def iii_rhs(self, i: int, x: np.ndarray) -> np.ndarray:
        """
        r.h.s. of condition (iii)
        """
        return (1 - self.alpha) * self.p1(i, x)
    

    def vii_lhs(self, i: int, x: np.ndarray) -> np.ndarray:
        """
        l.h.s. of condition (vii)
        """
        return 2 * self.p0(i, x) +  x * self.sum(0, i - self.m, x) + x * self.sum(1, i - self.m, x)


    def vii_rhs(self, i: int, x: np.ndarray) -> np.ndarray:
        """
        r.h.s. of condition (vii)
        """
        return (1 - self.alpha) * self.p1(i, x)
    

class num_bound:
    def __init__(self, lambda_f: float, delta_t: float, alpha: float, n: int, m: int) -> None:
        self.x = lambda_f * delta_t
        self.aa = 1.0 / (1.0 - alpha)
        self.n = n
        self.m = m
        self.eepsilon = np.zeros(self.n + 1, dtype=float)
        self.ee = np.zeros(self.n + 1, dtype=float)
    
    
    def load_data(self, epsilon: np.ndarray, e: np.ndarray) -> None:
        assert epsilon.shape[0] == self.n, 'Wrong dimension of epsilon'
        assert e.shape[0] == self.n, 'Wrong dimension of e'
        self.epsilon = epsilon
        self.e = e


    def fill_error(self):
        for k in range(self.n - 1, -1, -1):
            sum_init = k + 1
            sum_term = min(k + self.m - 1, self.n - 1)
            self.eepsilon[k] = (1.0 + self.x) * self.eepsilon[k + 1] + self.epsilon[k] + self.x * self.ee[k + 1]
            self.ee[k] = self.e[k] + self.aa * (self.eepsilon[min(k + self.m, self.n)] + self.eepsilon[k])
            if sum_init <= sum_term:
                self.ee[k] += self.aa * self.x * (np.sum(self.eepsilon[sum_init: sum_term + 1]) + np.sum(self.ee[sum_init: sum_term + 1]))





if __name__ == '__main__':
    alpha = 0.75
    m = 6
    plot_range = 10
    p = error_poly(alpha, m)
    x = np.arange(0, plot_range, 0.001, dtype=float)
    import matplotlib.pyplot as plt
    i = 8
    plt.plot(x, p.vii_rhs(i, x) - p.vii_lhs(i, x)) # iii is true if positive
    plt.xlim(0, plot_range)
    plt.show()
