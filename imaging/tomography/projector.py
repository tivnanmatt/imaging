
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
                print('Forward Projecting View : ', iTheta+1, ' of ', self.Ntheta)

            # compute the overlap between the voxel trapezoidal footprint and the pixel only for nonzero pixels. Return pixel indices and nonzero areas
            pixel_index, area_between_pixel_trapezoidal_footprint = self._system_response(theta_i, y2d, x2d)
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
                print('Back Projecting View : ', iTheta+1, ' of ', self.Ntheta)

            # compute the overlap between the voxel trapezoidal footprint and the pixel only for nonzero pixels. Return pixel indices and nonzero areas
            pixel_index, area_between_pixel_trapezoidal_footprint = self._system_response(theta_i, y2d, x2d)
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

    def _system_response(self, theta_i, y2d, x2d):

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
            # area of the center of the trapezoidm
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

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # compute the projection
    Nx = 128
    Ny = 128
    dx = 320.0/Nx
    dy = 320.0/Ny
    Nu = np.ceil(np.sqrt(Nx**2.0 + Ny**2.0)*10/8).astype(int) + 1
    du = dx*.8

    Ntheta = 210
    theta = np.linspace(0, np.pi, Ntheta)
    myProjector = CTProjector_ParallelBeam2D( Nx, Ny, dx, dy, Nu, du, theta, verbose=True).to(device)
    # myProjector._make_SRT()
    img = torch.zeros([Nx,Ny], dtype=torch.float32).to(device)
    # img[np.floor(Ny/2).astype(np.int32):np.ceil(Ny*.7).astype(np.int32), np.floor(Nx/2).astype(np.int32):np.ceil(Nx*.6).astype(np.int32)] = 1.0
    img[60:84,60:84] = 1.0
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