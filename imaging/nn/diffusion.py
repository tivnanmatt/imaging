import torch 

from .models import ConvolutionalMLP, Unet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class DiffusionModel(torch.nn.Module):
    def __init__(self,
                alpha_bar_func,
                time_decoder_base_channels=64,
                unet_base_channels=64,
                unet_in_channels=32,
                image_size=64,
                periodic_pad=False,
                 ):
        
        super(DiffusionModel, self).__init__()
        self.alpha_bar_func = alpha_bar_func
        self.time_decoder = ConvolutionalMLP(   base_channels=time_decoder_base_channels, 
                                                in_channels=1, 
                                                out_channels=unet_in_channels-2).to(device)
        self.unet = Unet(   base_channels=unet_base_channels, 
                            in_channels=unet_in_channels, 
                            out_channels=1,
                            periodic_pad=periodic_pad).to(device)
        self.optimizer = torch.optim.Adam(  self.parameters(), 
                                            lr=1e-3)
        self.image_size = image_size

    def forward(    self, 
                    x_t, 
                    y, 
                    t
                    ):
        t_dec = self.time_decoder(t).repeat(1, 1, self.image_size, self.image_size)
        x_t_and_y_and_t_dec = torch.cat([x_t, y, t_dec], dim=1)
        epsilon_hat = self.unet(x_t_and_y_and_t_dec)
        return epsilon_hat
    
    def train(  self, 
                x, 
                y, 
                num_epochs=10, 
                batch_size=128, 
                verbose=True
                ):
        num_batches = x.shape[0] // batch_size
        for iEpoch in range(num_epochs):
            for iBatch in range(num_batches):
                x_batch = x[iBatch*batch_size:(iBatch+1)*batch_size]
                y_batch = y[iBatch*batch_size:(iBatch+1)*batch_size]
                loss = self.train_step(x_batch, y_batch)
                if verbose:
                    print("Epoch: ", iEpoch, " Batch: ", iBatch, " Loss: ", loss)
    
    def train_step( self, 
                    x, 
                    y):
        # sample random times (different times for each batch element)
        t = torch.rand((x.shape[0], 1, 1, 1)).to(device)
        # sample random noise, normal distribution, mean=0, std=1
        epsilon = torch.randn((x.shape[0], 1, self.image_size, self.image_size)).to(device)
        # this function defines the forward process as a function of x_0, t, and epsilon
        x_t = self.sample_x_t_given_x_0_and_t_and_epsilon(x, t, epsilon)
        # now estimate epsilon given x_t, y, t
        epsilon_hat = self.forward(x_t, y, t)
        # compare the predicted epsilon to the actual epsilon
        loss = torch.mean((epsilon_hat - epsilon)**2)
        # backpropagate and update the weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def forward_process(self, 
                        x_0, 
                        batch_size=1, 
                        num_steps=128):
        # start time is always 0.0
        t = torch.zeros((batch_size, 1, 1, 1)).float().to(device)
        # max time is always 1.0, so dt is 1.0/num_steps
        dt = 1.0/num_steps
        # initialize x_t to the forward process input x_0
        x_t = x_0
        # initialize a tensor to hold all the x_t's
        x_t_all = torch.zeros((num_steps, batch_size, 1, self.image_size, self.image_size)).to(device)
        for i in range(num_steps):
            # sample next time
            x_t = self.sample_x_t_plus_dt_given_x_t_and_t_and_dt(x_t, t, dt)
            # update t
            t = t + dt
            # add to the list
            x_t_all[i] = x_t 
        return x_t_all
    
    def reverse_process(self, 
                        y,
                        x_T=None,
                        batch_size=1, 
                        num_steps=128,
                        returnFullProcess=True
                        ):
        t = torch.ones((batch_size, 1, 1, 1)).float().to(device)
        dt = 1.0/num_steps
        if x_T is None:
            x_t = self.sample_x_T_given_y(y)
        else:
            x_t = x_T
        if returnFullProcess:
            x_t_all = torch.zeros((num_steps, batch_size, 1, self.image_size, self.image_size)).to(device)
        for i in range(num_steps):
            # set up the unet and time decoder for this evaluation
            self.unet.eval()
            self.time_decoder.eval()
            # make a prediction
            with torch.no_grad():
                epsilon_hat = self.forward(x_t, y, t)
            # set the model back to training mode
            self.unet.train()
            self.time_decoder.train()
            x_t = self.sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_epsilon_hat(x_t, t, dt, epsilon_hat)
            # update t
            t = t - dt
            if returnFullProcess:
                # add to the list
                x_t_all[i] = x_t 

        if returnFullProcess:
            return x_t_all
        else:
            return x_t
    
    def sample_x_t_given_x_0_and_t_and_epsilon( self, 
                                                x_0, 
                                                t, 
                                                epsilon
                                                ):
        # this is needed for training
        alpha_bar = self.alpha_bar_func(t)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar)*epsilon
        return x_t
    
    def sample_x_t_plus_dt_given_x_t_and_t_and_dt(  self, 
                                                    x_t, 
                                                    t, 
                                                    dt
                                                    ):
        # this is needed for running the forward process
        alpha_bar = self.alpha_bar_func(t)
        alpha =  self.alpha_bar_func(t+dt) /alpha_bar
        # update x_t
        x_t_plus_dt = torch.sqrt(alpha)*x_t 
        x_t_plus_dt = x_t_plus_dt + torch.sqrt(1-alpha) * torch.randn_like(x_t)
        return x_t_plus_dt
    
    def sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_epsilon_hat( self, 
                                                                    x_t, 
                                                                    t, 
                                                                    dt, 
                                                                    epsilon_hat
                                                                    ):
        # this is needed for running the reverse process
        alpha_bar = self.alpha_bar_func(t)
        alpha = alpha_bar/self.alpha_bar_func(t-dt)
        # update x_t
        x_t_minus_dt = (x_t - ((1-alpha)/torch.sqrt(1-alpha_bar)) * epsilon_hat)*(1/torch.sqrt(alpha))
        x_t_minus_dt  = x_t_minus_dt + torch.sqrt(1-alpha) * torch.randn_like(x_t)
        return x_t_minus_dt
    
    def sample_x_T_given_y(self, y):
        return torch.randn((y.shape[0], 1, self.image_size, self.image_size)).to(device)












class ScalarDiffusionModel(DiffusionModel):
    def __init__(self,  signal_magnitude_func,
                        signal_magnitude_derivative_func,
                        noise_variance_func,
                        noise_variance_derivative_func,
                         **kwargs):
        def alpha_bar_func(t):
            return signal_magnitude_derivative_func(t)**2.0
        super(ScalarDiffusionModel, self).__init__(alpha_bar_func,**kwargs)
        self.signal_magnitude_func = signal_magnitude_func
        self.signal_magnitude_derivative_func = signal_magnitude_derivative_func
        self.noise_variance_func = noise_variance_func
        self.noise_variance_derivative_func = noise_variance_derivative_func
    
    def sample_x_t_given_x_0_and_t_and_epsilon( self, 
                                                x_0, 
                                                t, 
                                                epsilon
                                                ):

        # this is needed for training

        x_t = self.signal_magnitude_func(t) * x_0 + torch.sqrt(self.noise_variance_func(t)) * epsilon

        return x_t
    
    def sample_x_t_plus_dt_given_x_t_and_t_and_dt(  self, 
                                                    x_t, 
                                                    t, 
                                                    dt
                                                    ):

        # this is needed for running the forward process

        signal_magnitude_t = self.signal_magnitude_func(t)
        signal_magnitude_derivative_t = self.signal_magnitude_derivative_func(t)
        noise_variance_t = self.noise_variance_func(t)
        noise_variance_derivative_t = self.noise_variance_derivative_func(t)

        f = (signal_magnitude_derivative_t/signal_magnitude_t)*x_t
        g = torch.sqrt(-2*(signal_magnitude_derivative_t/signal_magnitude_t)*noise_variance_t + noise_variance_derivative_t)
        dw = torch.sqrt(torch.tensor(dt))*torch.randn_like(x_t)

        dx = f*dt + g*dw

        x_t_plus_dt = x_t + dx
        
        return x_t_plus_dt

    def sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_epsilon_hat(self, x_t, t, dt, epsilon_hat):

        # this is needed for running the reverse process

        signal_magnitude_t = self.signal_magnitude_func(t)
        signal_magnitude_derivative_t = self.signal_magnitude_derivative_func(t)
        noise_variance_t = self.noise_variance_func(t)
        noise_variance_derivative_t = self.noise_variance_derivative_func(t)

        f = (signal_magnitude_derivative_t/signal_magnitude_t)*x_t
        g = torch.sqrt(-2*(signal_magnitude_derivative_t/signal_magnitude_t)*noise_variance_t + noise_variance_derivative_t)
        score = -epsilon_hat*torch.sqrt(1/noise_variance_t)
        dw = torch.sqrt(torch.tensor(dt))*torch.randn_like(x_t)

        dx = (f - g*g*score)*dt + g*dw
        x_t_minus_dt = x_t - dx

        return x_t_minus_dt
    
    def sample_x_T_given_y(self, y):
        return torch.sqrt(self.noise_variance_func(1+ 0*y))*torch.randn((y.shape[0], 1, self.image_size, self.image_size)).to(device)









# funny....the code is exactly the same as ScalarDiffusionModel because of broadcasting...
class DiagonalDiffusionModel(DiffusionModel):
    def __init__(self,  signal_magnitude_func,
                        signal_magnitude_derivative_func,
                        noise_variance_func,
                        noise_variance_derivative_func,
                         **kwargs):
        def alpha_bar_func(t):
            return signal_magnitude_derivative_func(t)**2.0
        super(DiagonalDiffusionModel, self).__init__(alpha_bar_func,**kwargs)
        self.signal_magnitude_func = signal_magnitude_func
        self.signal_magnitude_derivative_func = signal_magnitude_derivative_func
        self.noise_variance_func = noise_variance_func
        self.noise_variance_derivative_func = noise_variance_derivative_func
    
    def sample_x_t_given_x_0_and_t_and_epsilon( self, 
                                                x_0, 
                                                t, 
                                                epsilon
                                                ):

        # this is needed for training

        x_t = self.signal_magnitude_func(t) * x_0 + torch.sqrt(self.noise_variance_func(t)) * epsilon
        
        return x_t
    
    def sample_x_t_plus_dt_given_x_t_and_t_and_dt(  self, 
                                                    x_t, 
                                                    t, 
                                                    dt
                                                    ):

        # this is needed for running the forward process

        signal_magnitude_t = self.signal_magnitude_func(t)
        signal_magnitude_derivative_t = self.signal_magnitude_derivative_func(t)
        noise_variance_t = self.noise_variance_func(t)
        noise_variance_derivative_t = self.noise_variance_derivative_func(t)

        f = (signal_magnitude_derivative_t/signal_magnitude_t)*x_t
        g = torch.sqrt(-2*(signal_magnitude_derivative_t/signal_magnitude_t)*noise_variance_t + noise_variance_derivative_t)
        dw = torch.sqrt(torch.tensor(dt))*torch.randn_like(x_t)

        dx = f*dt + g*dw

        x_t_plus_dt = x_t + dx
        
        return x_t_plus_dt

    def sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_epsilon_hat(self, x_t, t, dt, epsilon_hat):

        # this is needed for running the reverse process

        signal_magnitude_t = self.signal_magnitude_func(t)
        signal_magnitude_derivative_t = self.signal_magnitude_derivative_func(t)
        noise_variance_t = self.noise_variance_func(t)
        noise_variance_derivative_t = self.noise_variance_derivative_func(t)

        f = (signal_magnitude_derivative_t/signal_magnitude_t)*x_t
        g = torch.sqrt(-2*(signal_magnitude_derivative_t/signal_magnitude_t)*noise_variance_t + noise_variance_derivative_t)
        score = -epsilon_hat*torch.sqrt(1/noise_variance_t)
        dw = torch.sqrt(torch.tensor(dt))*torch.randn_like(x_t)

        dx = (f - g*g*score)*dt + g*dw
        x_t_minus_dt = x_t - dx

        return x_t_minus_dt
    
    def sample_x_T_given_y(self, y):
        return torch.sqrt(self.noise_variance_func(1+ 0*y))*torch.randn((y.shape[0], 1, self.image_size, self.image_size)).to(device)











class FourierDiffusionModel(DiffusionModel):
    def __init__(self,  MTF_func,
                        MTF_derivative_func,
                        NPS_func,
                        NPS_derivative_func,
                         **kwargs
                        ):
        def alpha_bar_func(t):
            return 0*t
        super(FourierDiffusionModel, self).__init__(alpha_bar_func,**kwargs)
        self.MTF_func = MTF_func
        self.MTF_derivative_func = MTF_derivative_func
        self.NPS_func = NPS_func
        self.NPS_derivative_func = NPS_derivative_func
    
    def sample_x_t_given_x_0_and_t_and_epsilon( self, 
                                                x_0, 
                                                t, 
                                                epsilon
                                                ):

        # this is needed for training

        x_0_fft = torch.fft.fft2(x_0, dim=(-2,-1))
        epsilon_fft = torch.fft.fft2(epsilon, dim=(-2,-1))

        x_t_fft = self.MTF_func(t)*x_0_fft + torch.sqrt(self.NPS_func(t))*epsilon_fft
        x_t = torch.fft.ifft2(x_t_fft, dim=(-2,-1)).real

        return x_t
    
    def sample_x_t_plus_dt_given_x_t_and_t_and_dt(  self, 
                                                    x_t, 
                                                    t, 
                                                    dt
                                                    ):

        # this is needed for running the forward process

        MTF_t = self.MTF_func(t)
        MTF_derivative_t = self.MTF_derivative_func(t)
        NPS_t = self.NPS_func(t)
        NPS_derivative_t = self.NPS_derivative_func(t)
        
        f = torch.fft.ifft2((MTF_derivative_t/MTF_t)*torch.fft.fft2(x_t, dim=(-2,-1)), dim=(-2,-1)).real
        g_transfer_function = torch.sqrt(-2*(MTF_derivative_t/MTF_t)*NPS_t + NPS_derivative_t)
        dw = torch.sqrt(torch.tensor(dt))*torch.randn_like(x_t)
        g_dw = torch.fft.ifft2(g_transfer_function*torch.fft.fft2(dw, dim=(-2,-1)), dim=(-2,-1)).real

        dx = f*dt + g_dw

        x_t_plus_dt = x_t + dx

        return x_t_plus_dt

    def sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_epsilon_hat(self, x_t, t, dt, epsilon_hat):

        # this is needed for running the reverse process

        MTF_t = self.MTF_func(t)
        MTF_derivative_t = self.MTF_derivative_func(t)
        NPS_t = self.NPS_func(t)
        NPS_derivative_t = self.NPS_derivative_func(t)

        f = torch.fft.ifft2((MTF_derivative_t/MTF_t)*torch.fft.fft2(x_t, dim=(-2,-1)), dim=(-2,-1)).real
        tmp = -2*(MTF_derivative_t/MTF_t)*NPS_t + NPS_derivative_t
        # hacky clip.... decide how to deal with this later...probably need to check for invalid processes
        tmp[tmp<1e-12] = 1e-12
        g_transfer_function = torch.sqrt(tmp)
        dw = torch.sqrt(torch.tensor(dt))*torch.randn_like(x_t)
        g_dw = torch.fft.ifft2(g_transfer_function*torch.fft.fft2(dw, dim=(-2,-1)), dim=(-2,-1)).real


        # f = torch.fft.ifft2((MTF_derivative_t/MTF_t)*torch.fft.fft2(x_t, dim=(-2,-1)), dim=(-2,-1)).real
        # g_transfer_function = torch.sqrt(-2*(MTF_derivative_t/MTF_t)*NPS_t + NPS_derivative_t)
        # dw = torch.sqrt(torch.tensor(dt))*torch.randn_like(x_t)
        # g_dw = torch.fft.ifft2(g_transfer_function*torch.fft.fft2(dw, dim=(-2,-1)), dim=(-2,-1)).real
        score = -torch.fft.ifft2(torch.sqrt(1/NPS_t)*torch.fft.fft2(epsilon_hat, dim=(-2,-1)), dim=(-2,-1)).real
        g2_score = torch.fft.ifft2(g_transfer_function*g_transfer_function*torch.fft.fft2(score, dim=(-2,-1)), dim=(-2,-1)).real
        
        dx = (f - g2_score)*dt + g_dw

        x_t_plus_dt = x_t - dx
        
        return x_t_plus_dt

    def sample_x_T_given_y(self, y):

        return y
    
    # def sample_x_T_given_y(self, y):

    #     x_0 = y
    #     epsilon = torch.randn_like(y)
    #     t = torch.ones((y.shape[0], 1, 1, 1)).to(device)

    #     x_T = self.sample_x_t_given_x_0_and_t_and_epsilon(x_0, t, epsilon)

    #     return x_T


























# def run_forward_process(ddpm,
#                         x,
#                         num_steps=1024,
#                         scale_for_plot=1,
#                         offset_for_plot=0,
#                         clim_for_plot=None,
#                         animation_filename=None,
#                         animation_subsample_factor=1,
#                         animation_fps=30):
   
#     x_t_forward_process = ddpm.forward_process(x, batch_size=x.shape[0], num_steps=num_steps)

#     if animation_filename is not None:

#         make_animation( x_t_forward_process[::animation_subsample_factor,0,0].cpu().detach().numpy()*scale_for_plot + offset_for_plot,
#                             clim=clim_for_plot,
#                             animation_filename=animation_filename,
#                             animation_fps=animation_fps,
#                             animation_title='Forward Stochastic Process',
#                             animation_name='Forward Process Animation')

#     return x_t_forward_process


# def run_reverse_process(ddpm,
#                         y,
#                         num_steps=1024,
#                         scale_for_plot=1,
#                         offset_for_plot=0,
#                         clim_for_plot=None,
#                         reverse_process_animation_filename=None,
#                         reverse_process_animation_fps=30,
#                         reverse_process_animation_subsample_factor=1,
#                         posterior_samples_animation_filename=None,
#                         posterior_samples_animation_fps=1):

#     x_t_reverse_process = ddpm.reverse_process(y, batch_size=y.shape[0], num_steps=num_steps)

#     if reverse_process_animation_filename is not None:

#         make_animation( x_t_reverse_process[::reverse_process_animation_subsample_factor,0,0].cpu().detach().numpy()*scale_for_plot + offset_for_plot,
#                             clim=clim_for_plot,
#                             animation_filename=reverse_process_animation_filename,
#                             animation_fps=reverse_process_animation_fps,
#                             animation_title='Reverse Stochastic Process',
#                             animation_name='Reverse Process Animation')

#     if posterior_samples_animation_filename is not None:

#         make_animation( x_t_reverse_process[-1,:,0].cpu().detach().numpy()*scale_for_plot + offset_for_plot,
#                             clim=clim_for_plot,
#                             animation_filename=posterior_samples_animation_filename,
#                             animation_fps=posterior_samples_animation_fps,
#                             animation_title='Posterior Samples',
#                             animation_name='Posterior Sampling Animation')

#     return x_t_reverse_process





# class DDPM_MTF_NPS(DDPM):
#     def __init__(self, MTF_func, NPS_func, **kwargs):
#         super(DDPM_MTF_NPS, self).__init__(**kwargs)
#         self.MTF_func = MTF_func
#         self.NPS_func = NPS_func

#     def sample_x_t_given_x_0_and_t_and_epsilon(self, x_0, t, epsilon):
#         # this is needed for training
#         nBatch = x_0.shape[0]
#         MTF = self.MTF_func(t).reshape((nBatch, 1, 64, 64))
#         NPS = self.NPS_func(t).reshape((nBatch, 1, 64, 64))
#         # take the fourier transform of x_0 on dimensions 2 and 3
#         x_0_fft = torch.fft.fft2(x_0, dim=(2,3))
#         epsilon_fft = torch.fft.fft2(epsilon, dim=(2,3))
#         # multiply by the MTF in the spatial frequency domain
#         x_t_fft = MTF * x_0_fft
#         # add the stationary noise in the spatial frequency domain
#         x_t_fft = x_t_fft + torch.sqrt(NPS) * epsilon_fft
#         # take the inverse fourier transform
#         x_t = torch.fft.ifft2(x_t_fft, dim=(2,3))
#         return x_t.real
    
#     def sample_x_t_plus_dt_given_x_t_and_t_and_dt(self, x_t, t, dt):
#         # this is needed for running the forward process

#         MTF_t = self.MTF_func(t).reshape((1, 1, 64, 64))
#         MTF_t_plus_dt = self.MTF_func(t+dt).reshape((1, 1, 64, 64))
#         NPS_t = self.NPS_func(t).reshape((1, 1, 64, 64))
#         NPS_t_plus_dt = self.NPS_func(t+dt).reshape((1, 1, 64, 64))

#         # compute the LSI system applied at this time step
#         H = MTF_t_plus_dt/MTF_t
#         # compute the stationary noise added at this time step
#         delta_NPS = NPS_t_plus_dt - NPS_t*H*H

#         # update x_t
#         x_t_fft = torch.fft.fft2(x_t, dim=(2,3))
#         x_t_plus_dt_fft = H * x_t_fft
#         x_t_plus_dt_fft = x_t_plus_dt_fft + torch.sqrt(delta_NPS) * torch.fft.fft2(torch.randn_like(x_t_fft), dim=(2,3))
#         x_t_plus_dt = torch.fft.ifft2(x_t_plus_dt_fft, dim=(2,3))

#         return x_t_plus_dt.real
    
#     def sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_epsilon_hat(self, x_t, t, dt, epsilon_hat):

#         nBatch = x_t.shape[0]

#         MTF_t = self.MTF_func(t).reshape((nBatch, 1, 64, 64))
#         MTF_t_minus_dt = self.MTF_func(t-dt).reshape((nBatch, 1, 64, 64))
#         NPS_t = self.NPS_func(t).reshape((nBatch, 1, 64, 64))
#         NPS_t_minus_dt = self.NPS_func(t-dt).reshape((nBatch, 1, 64, 64))

#         x_t_fft =torch.fft.fft2(x_t, dim=(2,3))
#         epsilon_hat_fft = torch.fft.fft2(epsilon_hat, dim=(2,3))
#         x_0_fft = x_t_fft - torch.sqrt(NPS_t) * epsilon_hat_fft
#         x_0_fft = x_0_fft/MTF_t

#         # compute the LSI system applied at this time step
#         H = MTF_t/MTF_t_minus_dt
#         # compute the stationary noise added at this time step
#         NPS_delta = NPS_t/H/H - NPS_t_minus_dt
#         NPS_posterior = 1/( (1/NPS_delta) + (1/NPS_t_minus_dt) )

#         x_t_minus_dt_fft = NPS_t_minus_dt * (1/H) * x_t_fft
#         x_t_minus_dt_fft += NPS_delta * MTF_t_minus_dt * x_0_fft
#         x_t_minus_dt_fft = x_t_minus_dt_fft / (NPS_delta + NPS_t_minus_dt)
#         x_t_minus_dt_fft += torch.sqrt(NPS_posterior) * torch.fft.fft2(torch.randn_like(x_t_fft), dim=(2,3))
#         x_t_minus_dt = torch.fft.ifft2(x_t_minus_dt_fft, dim=(2,3))

#         return x_t_minus_dt.real

