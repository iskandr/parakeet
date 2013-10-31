

import time 
import numpy as np 
from numpy import sqrt, int32, mod, double 
 
def calculate_distance(template, dx, dy) : 
    side_length = template.shape[0]
    # where is the center position
    cen = side_length/2
    
    for i in range(side_length) : 
        for j in range(side_length) : 
            template[i,j] = sqrt(((i-cen)*dx)**2 + ((j-cen)*dy)**2)




def kernel_loops(ds, h):
  fs = np.zeros_like(ds)
  denom = (np.pi*h**3)
  for i in xrange(ds.shape[0]):
    for j in xrange(ds.shape[1]):
      d = ds[i,j] 
      if d < 1 : 
        fs[i,j] = (1.-(3./2)*d**2 + (3./4.)*d**3) / denom 
      elif d <= 2.0 :
        fs[i,j] = 0.25*(2.-d)**3 / denom  
  return fs 
   


def physical_to_pixel(xpos,xmin,dx) : 
    return int32((xpos-xmin)/dx)

"""

def template_render_image(s,nx,ny,xmin,xmax,ymin,ymax,qty='rho',timing = False,two_d=0):
    
    time_init = time.clock()
    
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth',qty,'mass','rho']]

    # ----------------------
    # setup the global image
    # ----------------------
    image = np.zeros((nx,ny))
    
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny
    
    x_start = xmin+dx/2
    y_start = ymin+dy/2

    zplane = 0.0

    # ------------------------------------
    # trim particles based on image limits
    # ------------------------------------
    start = time.clock()
    ind = np.where((xs + 2*hs > xmin) & (xs - 2*hs < xmax) & 
                   (ys + 2*hs > ymin) & (ys - 2*hs < ymax) &
                   (np.abs(zs-zplane)*(1-two_d) < 2*hs))[0]

    xs,ys,zs,hs,qts,mass,rhos = (xs[ind],ys[ind],zs[ind],hs[ind],qts[ind],mass[ind],rhos[ind])
    if timing: print '<<< Initial particle selection took %f s'%(time.clock()-start)

    # set the render quantity 
    qts *= mass/rhos

    #
    # bin particles by how many pixels they need in their kernel
    #
    start = time.clock()
    npix = 2.0*hs/dx
    dbin = np.digitize(npix,np.arange(1,npix.max()))
    dbin_sortind = dbin.argsort()
    dbin_sorted = dbin[dbin_sortind]
    xs,ys,zs,hs,qts = (xs[dbin_sortind],ys[dbin_sortind],zs[dbin_sortind],hs[dbin_sortind],qts[dbin_sortind])
    if timing: print '<<< Bin sort done in %f'%(time.clock()-start)

    # ---------------------
    # process the particles 
    # ---------------------
    start = time.clock()
    image = template_kernel_cpu(xs,ys,qts,hs,nx,ny,xmin,xmax,ymin,ymax,two_d)
    if timing: print '<<< Rendering %d particles took %f s'%(len(xs),
                                                             time.clock()-start)
    
    if timing: print '<<< Total time: %f s'%(time.clock()-time_init)

    return image

"""

def template_kernel_cpu(xs,ys,qts,hs,nx,ny,xmin,xmax,ymin,ymax,two_d) : 
    # ------------------
    # create local image 
    # ------------------
    image = np.zeros((nx,ny),dtype=np.float)

    Npart = len(hs) 
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

       
    # ------------------------------
    # start the loop through kernels
    # ------------------------------
    kmax = int(np.ceil(hs.max()*2.0/dx*2.0))
    kmin = int(np.floor(hs.min()*2.0/dx*2.0))
    # make sure kmin and kmax are odd
    
    kmax += (1 - mod(kmax,2))
    kmin += (1 - mod(kmin,2))
    
    kmin = max(1,kmin)
    
    kernel_base = np.ones((kmax,kmax))
    kernel = np.ones((kmax,kmax))
    calculate_distance(kernel_base,dx,dy)
    
    max_d_curr = 0.0
    start_ind = 0
    end_ind = 0
    for k in xrange(kmin,kmax+2,2) : 
        # ---------------------------------
        # the max. distance for this kernel
        # ---------------------------------
        max_d_curr = dx*np.floor(k/2.0)
        if max_d_curr < dx/2.0 : 
          max_d_curr = dx/2.0

        i_max_d = double(1./max_d_curr)
        # -------------------------------------------------
        # find the chunk of particles that need this kernel
        # -------------------------------------------------
        end_ind = 0 
        for i in xrange(start_ind,Npart): 
            if 2*hs[end_ind] < max_d_curr:
              end_ind = i
            
        
        Nper_kernel = end_ind-start_ind
        
        # -------------------------------------------------------------------------
        # only continue with kernel generation if there are particles that need it!
        # -------------------------------------------------------------------------
        if Nper_kernel > 0 : 
            kernel = kernel_base[kmax/2-k/2:kmax/2+k/2+1,
                                 kmax/2-k/2:kmax/2+k/2+1]
            kernel = kernel_loops(kernel*i_max_d*2.0,1.0)
            kernel *= 8*i_max_d*i_max_d*i_max_d # kernel / h**3
        
            # --------------------------------
            # paint each particle on the image
            # --------------------------------
            for pind in xrange(start_ind,end_ind) : 
                x,y,h,qt = [xs[pind],ys[pind],hs[pind],qts[pind]]
                
                # set the minimum h to be equal to half pixel width
                #                h = max_d_curr*.5
                #h = max(h,0.55*dx)
                
                # particle pixel center
                xpos = physical_to_pixel(x,xmin,dx)
                ypos = physical_to_pixel(y,ymin,dy)
    
                left  = xpos-k/2
                upper = ypos-k/2

                for i in xrange(0,k) : 
                    for j in xrange(0,k): 
                        if ((i+left>=0) and (i+left < nx) and (j+upper >=0) and (j+upper<ny)) : 
                            image[(i+left),(j+upper)] += kernel[i,j]*qt


            start_ind = end_ind

    return image
  
N = 20
x = y = z = hs= qts = mass = rhos = np.random.rand(N)
nx=ny=100
args = (x, y, qts,hs, nx, ny, 0.0, 1.0, 0.0, 1.0,1)

template_kernel_cpu(*args)
from compare_perf import compare_perf        

compare_perf(template_kernel_cpu, args)
