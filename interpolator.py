from scipy import interpolate
def interpolate_grid(pi_samples, sample_points, ages, years, kx, ky):
    interpolator = interpolate.bisplrep(x=sample_points[:,0], y=sample_points[:,1], z=pi_samples, xb=ages[0], xe=ages[-1], yb=years[0], ye=years[-1], kx=kx, ky=ky)
    return interpolate.bisplev(x=ages, y=years, tck=interpolator)

