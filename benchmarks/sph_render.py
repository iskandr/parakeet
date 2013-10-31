"""
SPH renderer, modified from Numba version at:
https://gist.github.com/rokroskar/bdcf6c6b210ff0efc738#file-gistfile1-txt-L55
"""
 
 
import numpy as np
from numpy import int32
 
def kernel_func(d, h) : 
    if d < 1 : 
        f = 1.-(3./2)*d**2 + (3./4.)*d**3
    elif d<2 :
        f = 0.25*(2.-d)**3
    else :
        f = 0
    return f/(np.pi*h**3)
 
def distance(x,y,z) : 
    return np.sqrt(x**2+y**2+z**2)
 
def physical_to_pixel(xpos,xmin,dx) : 
    return int32((xpos-xmin)/dx)
 
def pixel_to_physical(xpix,x_start,dx) : 
    return dx*xpix+x_start
 
def render_image(xs, ys, zs, hs, qts, mass, rhos, nx, ny, 
                  xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0): 
    MAX_D_OVER_H = 2.0
 
    image = np.zeros((nx,ny))
 
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny
 
    x_start = xmin+dx/2
    y_start = ymin+dy/2
    zplane = 0.0
 
    # set up the kernel values
    kernel_samples = np.arange(0, 2.01, .01)
    kernel_vals = np.array([kernel_func(x,1.0) for x in kernel_samples])
    for i, (x,y,z,h) in enumerate(zip(xs,ys,zs,hs)):
        qt = qts[i] * mass[i] / rhos[i]

        # is the particle in the frame?
        if ((x > xmin-2*h) and (x < xmax+2*h) and 
            (y > ymin-2*h) and (y < ymax+2*h) and 
            (np.abs(z-zplane) < 2*h)) : 
        
                    
            if (MAX_D_OVER_H*h/dx < 1 ) and (MAX_D_OVER_H*h/dy < 1) : 
                # pixel coordinates 
                xpos = physical_to_pixel(x,xmin,dx)
                ypos = physical_to_pixel(y,ymin,dy)
                # physical coordinates of pixel
                xpixel = pixel_to_physical(xpos,x_start,dx)
                ypixel = pixel_to_physical(ypos,y_start,dy)
                zpixel = zplane
 
                dxpix, dypix, dzpix = [x-xpixel,y-ypixel,z-zpixel]
                d = distance(dxpix,dypix,dzpix)
                if (xpos > 0) and (xpos < nx) and (ypos > 0) and (ypos < ny) and (d/h < 2) : 
                    kernel_val = kernel_vals[int(d/(.01*h))]/(h*h*h)
                    image[xpos,ypos] += qt*kernel_val
 
            else :
                # bottom left of pixels the particle will contribute to
                x_pix_start = int32(physical_to_pixel(x-MAX_D_OVER_H*h,xmin,dx))
                x_pix_stop  = int32(physical_to_pixel(x+MAX_D_OVER_H*h,xmin,dx))
                y_pix_start = int32(physical_to_pixel(y-MAX_D_OVER_H*h,ymin,dy))
                y_pix_stop  = int32(physical_to_pixel(y+MAX_D_OVER_H*h,ymin,dy))
            
                if(x_pix_start < 0):  x_pix_start = 0
                if(x_pix_stop  > nx): x_pix_stop  = int32(nx-1)
                if(y_pix_start < 0):  y_pix_start = 0
                if(y_pix_stop  > ny): y_pix_stop  = int32(ny-1)
    
                
                for xpix in range(x_pix_start, x_pix_stop) : 
                    for ypix in range(y_pix_start, y_pix_stop) : 
                        # physical coordinates of pixel
                        xpixel = pixel_to_physical(xpix,x_start,dx)
                        ypixel = pixel_to_physical(ypix,y_start,dy)
                        zpixel = zplane
 
                        dxpix, dypix, dzpix = [x-xpixel,y-ypixel,z-zpixel]
                        d = distance(dxpix,dypix,dzpix)
                        if (d/h < 2) : 
                            kernel_val = kernel_vals[int(d/(.01*h))]/(h*h*h)
                            image[xpix,ypix] += qt*kernel_val
    
 
    return image

from compare_perf import compare_perf 

N = 1600
x = y = z = hs= qts = mass = rhos = np.random.rand(N)
nx=ny=40
args = (x,y,z,hs,qts,mass,rhos,nx,ny, 0.0, 1.0, 0.0, 1.0)
compare_perf(render_image, args)
