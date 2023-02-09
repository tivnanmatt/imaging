
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# make a custom torch module
class CTProjector_ParallelBeam2D(torch.nn.Module):

    def __init__(self, Nx, Ny, dx, dy, Nu, du, theta,verbose=False):
        super(CTProjector_ParallelBeam2D, self).__init__()

        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.Nu = Nu
        self.du = du
        self.theta = theta
        self.Ntheta = len(theta)
        self.verbose = verbose

    def forward(self, img):

        return self._forward_project(img)

    def backward(self, proj):

        return self._back_project(proj)

    def _forward_project(self,img):

        assert img.ndim == 2, 'image must be 2D but got %dD' % img.ndim

        # define a meshgrid for the projections
        # delta_u = torch.tensor([-1,0,1],device=device)* self.du
        y = torch.linspace(-(self.Ny-1)/2, (self.Ny-1)/2, self.Ny, device=device) * self.dy
        x = torch.linspace(-(self.Nx-1)/2, (self.Nx-1)/2, self.Nx, device=device) * self.dx
        
        # make a 2D meshgrid of (y, x) coordinates
        y2d, x2d = torch.meshgrid( y, x, indexing='ij')
        
        # initialize the projection
        proj = torch.zeros([self.Ntheta, self.Nu], dtype=torch.float32).to(device)

        # loop through thetas
        for iTheta, theta_i in enumerate(self.theta):
            if self.verbose:
                print('Forward Projecting View : ', iTheta, ' of ', self.Ntheta)

            # compute the overlap between the voxel trapezoidal footprint and the pixel only for nonzero pixels. Return pixel indices and nonzero areas
            pixel_index, area_between_pixel_trapezoidal_footprint = self._make_SRT_single_view_sparse(theta_i, y2d, x2d)
            num_nonzero_pixels = pixel_index.shape[0]

            # need to convert indices and areas to vectors for use with torch.Tensor.index_add_()
            pixel_index_vectors = pixel_index.reshape([num_nonzero_pixels, self.Nx*self.Ny])
            area_overlap_vectors = area_between_pixel_trapezoidal_footprint.reshape([num_nonzero_pixels, self.Nx*self.Ny])

            # image values as a vector
            img_vector = img.reshape([self.Nx*self.Ny])

            # loop through nonzero pixels
            for iPixel in range(num_nonzero_pixels):
                # multiply area of overlap by accumulate the forward projected volume into the projection
                proj[iTheta].index_add_(0, pixel_index_vectors[iPixel], img_vector*area_overlap_vectors[iPixel])
        return proj

    def _back_project(self,proj):

        assert proj.ndim == 2, 'projections must be 2D but got %dD' % proj.ndim

        # define a meshgrid for the projections
        # delta_u = torch.tensor([-1,0,1],device=device)* self.du
        y = torch.linspace(-(self.Ny-1)/2, (self.Ny-1)/2, self.Ny, device=device) * self.dy
        x = torch.linspace(-(self.Nx-1)/2, (self.Nx-1)/2, self.Nx, device=device) * self.dx
        
        # make a 2D meshgrid of (y, x) coordinates
        y2d, x2d = torch.meshgrid( y, x, indexing='ij')
        
        # initialize the projection
        img = torch.zeros([self.Ny, self.Nx], dtype=torch.float32).to(device)

        # loop through thetas
        for iTheta, theta_i in enumerate(self.theta):
            if self.verbose:
                print('Back Projecting View : ', iTheta, ' of ', self.Ntheta)

            # compute the overlap between the voxel trapezoidal footprint and the pixel only for nonzero pixels. Return pixel indices and nonzero areas
            pixel_index, area_between_pixel_trapezoidal_footprint = self._make_SRT_single_view_sparse(theta_i, y2d, x2d)
            num_nonzero_pixels = pixel_index.shape[0]

            # loop through nonzero pixels
            for iPixel in range(num_nonzero_pixels):
                # multiply area of overlap by accumulate the forward projected volume into the projection
                img += proj[iTheta, pixel_index[iPixel]]*area_between_pixel_trapezoidal_footprint[iPixel]
        return img

    def _convert_u_to_projection_index(self, u):
        return (u/self.du) + (self.Nu-1)/2.0

    def _convert_projection_index_to_u(self, projection_index):
        return (projection_index - (self.Nu-1)/2.0)*self.du

    def _make_SRT_single_view_sparse(self, theta_i, y2d, x2d):

        # compute the projection
        theta_i = torch.tensor(theta_i)

        # absolute value of the sine and cosine of the angle
        abs_cos_theta = torch.abs(torch.cos(theta_i))
        abs_sin_theta = torch.abs(torch.sin(theta_i))

        # height of the trapezoid
        h = torch.minimum(self.dy/abs_cos_theta, self.dx/abs_sin_theta)
        # base 1 of the trapezoid
        b1 = torch.abs(self.dy*abs_sin_theta - self.dx*abs_cos_theta)
        # base 2 of the trapezoid
        b2 = torch.abs(self.dy*abs_sin_theta + self.dx*abs_cos_theta)

        # below depends on x and y, above depends only on theta

        # center of the trapezoid
        u0 = (x2d*torch.cos(theta_i) + y2d*torch.sin(theta_i))
        # left edge of the trapezoid base 2 
        u1 = u0 - b2/2
        # left edge of the trapezoid base 1
        u2 = u0 - b1/2
        # right edge of the trapezoid base 1
        u3 = u0 + b1/2
        # right edge of the trapezoid base 2
        u4 = u0 + b2/2

        # compute the index of the projection
        u1_index = self._convert_u_to_projection_index(u1)
        u4_index = self._convert_u_to_projection_index(u4)
        num_nonzero_pixels = int(torch.max(torch.ceil(u4_index)-torch.floor(u1_index))) + 1
        pixel_index = torch.zeros([num_nonzero_pixels, x2d.shape[0], x2d.shape[1]], dtype=torch.long).to(device)
        area_between_pixel_trapezoidal_footprint = torch.zeros([num_nonzero_pixels, x2d.shape[0], x2d.shape[1]], dtype=torch.float).to(device)

        for iPixel in range(num_nonzero_pixels):
            # get the index of the pixel of interest
            pixel_index[iPixel] = torch.floor(u1_index).long() + iPixel
            # convert index to u coordinate
            u = self._convert_projection_index_to_u(pixel_index[iPixel])
            # area of the left side of the trapezoid
            u_A = torch.maximum(u1,u-self.du/2)
            u_B = torch.minimum(u2,u+self.du/2)
            area_between_pixel_trapezoidal_footprint[iPixel] += (u_B>u_A)*(h/(2*(u2-u1 + (u1==u2))))*((u_B-u1)**2.0 - (u_A-u1)**2.0)
            # area of the center of the trapezoid
            u_A = torch.maximum(u2,u-self.du/2)
            u_B = torch.minimum(u3,u+self.du/2)
            area_between_pixel_trapezoidal_footprint[iPixel] += (u_B>u_A)*h*(u_B-u_A)
            # area of the right side of the trapezoid
            u_A = torch.maximum(u3,u-self.du/2)
            u_B = torch.minimum(u4,u+self.du/2)
            area_between_pixel_trapezoidal_footprint[iPixel] += (u_B>u_A)*(h/(2*(u4-u3+ (u3==u4))))*((u_A-u4)**2.0 - (u_B-u4)**2.0)

        return pixel_index, area_between_pixel_trapezoidal_footprint 



if __name__ == '__main__':

    import numpy as np
    from matplotlib import pyplot as plt

    # compute the projection
    Nx = 512
    Ny = 512
    dx = 320.0/Nx
    dy = 320.0/Ny
    Nu = np.ceil(np.sqrt(Nx**2.0 + Ny**2.0)*10/8).astype(int) + 1
    du = dx*.8
    theta = np.linspace(0, np.pi, 500)
    myProjector = CTProjector_ParallelBeam2D( Nx, Ny, dx, dy, Nu, du, theta, verbose=True).to(device)
    # myProjector._make_SRT()
    img = torch.zeros([Nx,Ny], dtype=torch.float32).to(device)
    # img[np.floor(Ny/2).astype(np.int32):np.ceil(Ny*.7).astype(np.int32), np.floor(Nx/2).astype(np.int32):np.ceil(Nx*.6).astype(np.int32)] = 1.0
    img[100:104,100:104] = 1.0
    # proj = myProjector._apply_SRT(img)
    proj = myProjector.forward(img)

    ATAimg = myProjector.backward(proj)

    # plot the image and projection
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img.detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(proj.detach().cpu().numpy(), cmap='gray',aspect='auto')
    plt.colorbar()
    plt.show(block=True)

    # plot the image and projection
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img.detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(ATAimg.detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.show(block=True)




















# class CTProjector_ParallelBeam2D_(torch.nn.Module):
#     def __init__(   self, 
#                     image_size, 
#                     detector_size, 
#                     detector_spacing, 
#                     view_angles,
#                     pixels_per_voxel=3):
        
#         super(CTProjector_ParallelBeam2D_, self).__init__()

#         # keep it simple. inputs are ints and floats and lists thereof
#         assert type(image_size) == int
#         assert type(detector_size) == int
#         assert type(detector_spacing) == float
#         assert type(view_angles) == list
#         assert len(view_angles) > 0
#         for i in range(len(view_angles)):
#             assert type(view_angles[i]) == float
#         assert type(pixels_per_voxel) == int

#         self.image_size = image_size
#         self.detector_size = detector_size
#         self.detector_spacing = detector_spacing
#         self.view_angles = view_angles
#         self.pixels_per_voxel = pixels_per_voxel

#         self.nViews = len(view_angles)
#         self.Nx = image_size
#         self.Ny = image_size
#         self.Nu = detector_size

#         # make a meshgrid for the image space x,y
#         self.x_coord = torch.linspace(-(self.Nx-1)/2, (self.Nx-1)/2, self.Nx).to(device)
#         self.y_coord = torch.linspace(-(self.Ny-1)/2, (self.Ny-1)/2, self.Ny).to(device)
#         self.xGrid, self.yGrid = torch.meshgrid(self.x_coord, self.y_coord)

#         # make a meshgrid for the detector space u
#         self.u_coord = torch.linspace(-(self.Nu-1)/2, (self.Nu-1)/2, self.Nu).to(device)

#     def forward_project(self, volume):

#         assert volume.shape == (self.Nx, self.Ny)

#         # allocate the projection
#         projection = torch.zeros(self.nViews, self.Nu).to(device)

#         # loop over the views
#         for iView in range(self.nViews):
            
#             # angle of the view
#             phi = self.view_angles[iView]*torch.ones(self.Nx, self.Ny).to(device)
#             # absolute value of the sine and cosine of the angle
#             abs_cos_theta = torch.abs(torch.cos(phi))
#             abs_sin_theta = torch.abs(torch.sin(phi))
#             # height of the trapezoid
#             h = torch.minimum(1/abs_cos_theta, 1/abs_sin_theta)
#             # base 1 of the trapezoid
#             b1 = torch.abs(1*abs_sin_theta - 1*abs_cos_theta)
#             # base 2 of the trapezoid
#             b2 = abs_sin_theta + abs_cos_theta
#             # center of the trapezoid
#             u0 = self.xGrid*torch.cos(phi) + self.yGrid*torch.sin(phi)
#             # left edge of the trapezoid base 2 in the projection
#             u1 = u0 - b2/2.0
#             # left edge of the trapezoid base 1 in the projection
#             u2 = u0 - b1/2.0
#             # right edge of the trapezoid base 1 in the projection
#             u3 = u0 + b1/2.0
#             # right edge of the trapezoid base 2 in the projection
#             u4 = u0 + b2/2.0
            
#             # compute the index of the projection
#             u1_index = self._convert_u_to_projection_index(u1)
#             u4_index = self._convert_u_to_projection_index(u4)
#             num_nonzero_pixels = int(torch.max(torch.ceil(u4_index)-torch.floor(u1_index)))
#             for i in range(num_nonzero_pixels):
#                 # projection_index = torch.floor(u1_index) + i
#                 # u = self._convert_projection_index_to_u(projection_index)

#                 for iPixel in range(self.Nu):
#                     # projection coordinates
#                     # u = np.linspace(-10.0, 10.0, 100000)
#                     # initialize projection
#                     # proj = u*0
#                     # idx = (self.u_coord[iPixel]>u1) & (self.u_coord[iPixel]<u2)
#                     # projection[iView, iPixel] += h*((self.u_coord[iPixel]-u1)/(u2-u1))

#                 # # projection of the left side of the trapezoid
#                 # idx = (u>=u1) & (u<u2)
#                 # proj[idx] = h*((u[idx]-u1)/(u2-u1))
#                 # # projection of the central flat part fo the trapezoid
#                 # idx = (u>=u2) & (u<u3)
#                 # proj[idx] = h
#                 # # projection of the right side of the trapezoid
#                 # idx = (u>=u3) & (u<u4)
#                 # proj[idx] = h*(1.0  - (u[idx]-u3)/(u4-u3))    

#             print(num_nonzero_pixels)
            
#     def _convert_u_to_projection_index(self, u):
#         return (u/self.detector_spacing) + (self.Nu-1)/2.0
#     def _convert_projection_index_to_u(self, projection_index):
#         return (projection_index - (self.Nu-1)/2.0)*self.detector_spacing



#     def _forward_radon(self,img):

#         assert img.ndim == 2, 'image must be 2D but got %dD' % img.ndim

#         device = img.device 

#         # define a meshgrid for the projections
#         u = torch.linspace(-(self.Nu-1)/2, (self.Nu-1)/2, self.Nu, device=device) * self.du
#         y = torch.linspace(-(self.Ny-1)/2, (self.Ny-1)/2, self.Ny, device=device) * self.dy
#         x = torch.linspace(-(self.Nx-1)/2, (self.Nx-1)/2, self.Nx, device=device) * self.dx
        
#         # make a 3D meshgrid of (x,y,u) coordinates
#         u3d, y3d, x3d = torch.meshgrid(u, y, x, indexing='ij')

#         # initialize the projection
#         proj = torch.zeros([self.Ntheta, self.Nu], dtype=torch.float32)

#         # loop through thetas
#         for iTheta, theta in enumerate(self.theta):
#             print(iTheta)
#             A = self._make_SRT_single_view(theta, u3d, y3d, x3d)
#             for iU in range(self.Nu):
#                 # compute the projection
#                 proj[iTheta,iU] = torch.sum(img.reshape([1,img.shape[0],img.shape[1]])*A[iU])
                
#         return proj

