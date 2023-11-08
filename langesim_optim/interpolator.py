def interpolate(t, yi, yf, ti, tf, ylist, continuous=True):
    if t <= ti:
        return yi
    if t >= tf:
        return yf
    if continuous:
        # warning: self.ylist = [yi] + ylist + [yf] does not concatenate numpy arrays
        yl = [yi, *ylist, yf]
    else:
        yl = ylist
    N = len(yl)
    if N == 1:
        # TSP case: return one constant value
        y = yl[0]
    else:
        dt = (tf - ti) / (N - 1)
        idx = int( (t - ti) / dt )
        if idx >= N - 1:
            y = yl[N - 1]
        else:
            t1 = idx * dt + ti
            y = yl[idx] + (yl[idx + 1] - yl[idx]) * (t - t1) * dt**-1
    
    return y

class Interpolator():
    """Builds a linear interpolation function y(t) from a list of values ylist
    at times [ti, tf] and initial and final values yi, yf.
    """
    def __init__(self, yi, yf, ti, tf, ylist, continuous=True):
        """Initializes the interpolator function y(t) from a list of values ylist
        at times [ti, tf] and initial and final values yi, yf.

        Args: 
            yi: initial value of y
            yf: final value of y
            ti: initial time
            tf: final time
            ylist (list): list of values of y at times [ti, tf]
            continuous (bool): whether the interpolator is continuous at ti and tf
        """ 
        self.yi = yi
        self.yf = yf
        self.ti = ti
        self.tf = tf
        self.continuous = continuous
        self.ylist = ylist
        # if continuous:
        #     # warning: self.ylist = [yi] + ylist + [yf] does not concatenate numpy arrays
        #     self.ylist = [yi, *ylist, yf]
        # else:
        #     self.ylist = ylist
        # self.N = len(self.ylist)
        # if self.N > 1:
        #     self.dt = (self.tf - self.ti) / (self.N - 1)

    def __call__(self, t):
        return interpolate(t, self.yi, self.yf, self.ti, self.tf, self.ylist, self.continuous)
        # if t <= self.ti:
        #     return self.yi
        # if t >= self.tf:
        #     return self.yf
        
        # if self.N == 1:
        #     # TSP case: return one constant value
        #     y = self.ylist[0]
        # else:
        #     idx = int( (t - self.ti) / self.dt )
        #     if idx >= self.N - 1:
        #         y = self.ylist[N - 1]
        #     else:
        #         t1 = idx * self.dt + self.ti
        #         y = self.ylist[idx] + (self.ylist[idx + 1] - self.ylist[idx]) * (t - t1) * self.dt**-1
        
        # return y

def make_interpolator(yi, yf, ti, tf, ylist, continuous=True):
    """Obsolete: refactored as class Interpolator.
    
    Builds a linear interpolation function y(t) from a list of values ylist
    at times [ti, tf] and initial and final values yi, yf.

    Args: 
        yi: initial value of y
        yf: final value of y
        ti: initial time
        tf: final time
        ylist (list): list of values of y at times [ti, tf]
        continuous (bool): whether the interpolator is continuous at ti and tf
    """ 
    def interpolator(t, yi=yi, yf=yf, ti=ti, tf=tf, ylist=ylist, continuous=continuous):
        if t <= ti:
            return yi
        if t >= tf:
            return yf

        if continuous:
            yl = [yi, *ylist, yf]
        else:
            yl = ylist
    
        N = len(yl)
        if N == 1:
            # TSP case: only one constant value
            y = yl[0]
        else:
            dt = (tf - ti) / (N - 1)
            idx = int( (t - ti) / dt)
            if idx >= N - 1:
                y = yl[N - 1]
            else:
                t1 = idx * dt + ti
                y = yl[idx] + (yl[idx + 1] - yl[idx]) * (t - t1) * dt**-1
        
        return y
    
    return interpolator
