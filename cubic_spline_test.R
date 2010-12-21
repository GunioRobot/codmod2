# data
	data(engine, package="gamair")
	x <- (engine$size-min(engine$size))
	x <- x/max(x)
	wear <- engine$wear
	xk <- 1:7/8

# cubic spline basis (the "f(x1,x2)")
	rk <- function(x,z) {
		((z-.5)^2-1/12)*((x-.5)^2-1/12)/4 - ((abs(x-z)-.5)^4-(abs(x-z)-.5)^2/2+7/240)/24
	}

# produce model matrix for spline given x and set of knots
	spl.X <- function(x,xk) {
		q <- length(xk)+2	# number of parameters
		n <- length(x)		# number of datapoints
		X <- matrix(1,n,q)	# initialize model matrix
		X[,2] <- x			# set second column to x
		X[,3:q] <- outer(x,xk,FUN=rk)
							# set remaining to R(x,xk)
		X
	}
	
# find S (penalized regression spline matrix) given knot locations
	spl.S <- function(xk) {
		q <- length(xk)+2
		S <- matrix(0,q,q)	# initilize spline matrix
		S[3:q,3:q] <- outer(xk,xk,FUN=rk)
							# fill in with R(xk,xk)
		S
	}

# find the square root of the S matrix (ie B)
	mat.sqrt <- function(S) {
		d <- eigen(S,symmetric=TRUE)
		rS <- d$vectors %*% diag(sqrt(d$values)) %*% solve(d$vectors)
	}

# fit penalized regression spline
	prs.fit <- function(y,x,xk,lambda) {
		q <- length(xk)+2	# dimension of basis
		n <- length(x)		# number of datapoints
		Xa <- rbind(spl.X(x,xk), mat.sqrt(spl.S(xk))*sqrt(lambda))
							# create model matrix
		y[(n+1):(n+q)] <- 0	# augment data vector
		lm(y~Xa-1)			# fit penalized regression spline
	}

# run the model
	lambda <- .0001			# smoothing parameter
	xp <- seq(0,1,.01)		# prediction mesh
	plot(x,wear)

	xk <- 1:7/8				# knot locations
	Xp <- spl.X(xp,xk)		# prediction matrix
	lines(xp, Xp %*% coef(prs.fit(wear,x,xk,lambda=.0001)))









