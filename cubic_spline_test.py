# import data
import numpy
x = numpy.array([2.78, 2.43, 2.32, 2.43, 2.98, 2.32, 2.32, 2.32, 2.32, 2.13, 2.13, 2.13, 2.98, 1.99, 1.99, 1.99, 1.78, 1.58, 1.42])
x -= numpy.min(x)
x /= numpy.max(x)
y = numpy.array([3.0, 3.1, 4.8, 3.3, 2.8, 2.9, 3.8, 3.0, 2.7, 3.2, 2.4, 2.6, 1.7, 2.6, 2.8, 2.4, 2.5, 4.2, 4.0])

# create the basis function
def rk(x,z) :
	return numpy.array(((z-.5)**2.-1./12.)*((x-.5)**2.-1./12.)/4.-((abs(x-z)-.5)**4.-(abs(x-z)-.5)**2./2.+7./240.)/24.)

# produce model matrix for spline given x and set of knots
''' How to do this with outer product instead of enumerate? '''
def spl_X(x,xk) :
	X = numpy.ones((x.shape[0],xk.shape[0]+2.))
	X[:,1] = x
	for a,b in enumerate(x) :
		for c,d in enumerate(xk) :
			X[a,c+2] = rk(b,d)
	return X

# find S (penalized regression spline matrix) given knot locations
''' How to do this with outer product instead of enumerate? '''
def spl_S(xk) :
	S = numpy.zeros((xk.shape[0]+2.,xk.shape[0]+2.))
	for a,b in enumerate(xk) :
		for c,d in enumerate(xk) :
			S[a+2,c+2] = rk(b,d)
	return S
	
# find the square root of the S matrix (ie B)
def mat_sqrt(S) :
	d = numpy.linalg.eigh(S)
	vec = d[1][:,range(S.shape[0]-1,-1,-1)]
	val = d[0][range(S.shape[0]-1,-1,-1)]
	rS = numpy.dot(numpy.dot(vec,numpy.diag(numpy.sqrt(val))),numpy.linalg.solve(vec,numpy.identity(S.shape[0])))
	return rS

# fit penalized regression spline
def prs(y,x,xk,lam) :
	Xa = numpy.concatenate((spl_X(x,xk), mat_sqrt(spl_S(xk))*numpy.sqrt(lam)))
	y1 = numpy.concatenate((y,numpy.zeros(xk.shape[0]+2)))
	lst_sq = numpy.linalg.lstsq(Xa,y1)
	return lst_sq[0]

# make predictions
nk = 7													# number of knots
xk = numpy.array([k/(nk+1.) for k in range(1,nk+1)])	# locations of knots
lam = .0001												# smoothing parameter
xp = numpy.array([p/(100.) for p in range(0,101)])		# prediction mesh
yp = numpy.dot(spl_X(xp,xk), prs(y,x,xk,lam))

# plot predictions
import pylab
pylab.plot(xp,yp,'b-',x,y,'ro')











