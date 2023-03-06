
import torch
from torch import nn

from scipy import io

from xraydb import mu_elam 
from xraydb import material_mu


# load TASMICS data which describes bremsstrahlung spectrum
# hacky solution but works for now
import pkgutil
data_bytes = pkgutil.get_data('imaging', 'data/TASMICSdata.mat')
# write the data to a temporary file called TASMICSDATA.mat
with open('TASMICSdata.mat', 'wb') as f:
    f.write(data_bytes)
spTASMICS = io.loadmat('TASMICSdata.mat')['spTASMICS']
# delete the temporary file
import os
os.remove('TASMICSdata.mat')

# use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# we always assume 150 energy bins from 1 keV to 150 keV
nEnergy = 150
photon_kev = torch.linspace(1, nEnergy, nEnergy).to(device)

# mass attenuation spectra for each element. note we leave element 0 as 
# IMPORTANT: we use mm units for all lengths and mg units for all masses, so mass attenuation coefficient units are mm2/mg
massAttenuationSpectra = torch.zeros([nEnergy,91]).to(device)
for z in range(1,91):
    massAttenuationSpectra[:,z] =  torch.tensor(mu_elam(z, photon_kev.cpu()*1000, kind='total')) # native output of xraydb.mu_elam is cm2/g
    massAttenuationSpectra[:,z] = massAttenuationSpectra[:,z]*100 # mm2/g
    massAttenuationSpectra[:,z] = massAttenuationSpectra[:,z]*1e-3 # mm2/mg


# compute the source spectrum for a given (list of) kvp value(s)
def get_TASMICS_Spectrum(kvp: torch.Tensor) -> torch.Tensor:
    """
    Calculates the spectrum of an x-ray source using TASMICS.
    
    This method takes the peak voltage of the x-ray source as input and returns the corresponding spectrum using the TASMICS method. The input can be a single value or a batch of values. The output has the same shape as the input, with a spectrum calculated for each element in the batch.
    
    Args:
        kvp (torch.Tensor): The peak voltage of the xray source. Shape: [nBatch] or []. Type: torch.float32. Description: The peak voltage of the xray source. Batches are computed independently and in parallel.
        
    Returns:
        torch.Tensor: The spectrum of the xray source. Shape: [nBatch, 150]. Type: torch.float32. Description: The spectrum of the xray source. For each batch element, there is one spectrum.
        
    Example:
        >>> import torch
        >>> import miel
        >>> device = torch.device('cuda:0')
        >>> source_spectrum = miel.spectral.get_TASMICS_Spectrum(torch.tensor(120.).to(device))
        >>> print(source_spectrum.shape)
        >>> source_spectrum = miel.spectral.get_TASMICS_Spectrum(torch.tensor([100., 120., 140.]).to(device))
        >>> print(source_spectrum.shape)
    """

    assert torch.is_tensor(kvp), 'kvp must be a torch tensor'
    assert kvp.dtype == torch.float32, 'kvp must be a torch.float32 tensor'
    assert kvp.ndim == 1 or kvp.ndim == 0, 'kvp must be a 1D tensor or a scalar'

    noBatchFlag = False
    # handle the [] by reshaping to [1]
    if kvp.shape == torch.Size([]):
        noBatchFlag = True
        kvp = kvp.reshape(1)

    # initialize source spectrum
    source_spectrum = torch.zeros([kvp.shape[0], 150]).to(device)

    # loop over batches
    for iBatch, kvp_ in enumerate(kvp):

        if not torch.is_tensor(kvp_):
            kvp_ = torch.tensor(kvp_)
            
        assert(kvp_ >= 20 and kvp_ <= 140), 'kvp must be between 20 and 140'

        # define source spectrum
        kvp_floor = torch.floor(kvp_)
        kvp_ceil = torch.ceil(kvp_)
        if kvp_ceil == kvp_floor:
            sourceSpectrum = torch.tensor(spTASMICS[:,int(kvp_floor-20)])
        else:
            sourceSpectrum = (kvp_ceil - kvp_)*torch.tensor(spTASMICS[:,int(kvp_floor-20)]) + (kvp_ - kvp_floor)*torch.tensor(spTASMICS[:,int(kvp_ceil-20)])
        
        sourceSpectrum = sourceSpectrum/torch.sum(sourceSpectrum)

        source_spectrum[iBatch,:] = sourceSpectrum

        if noBatchFlag:
            source_spectrum = source_spectrum[0,:]

    return source_spectrum






class SpectralAttenuator(nn.Module):
    def __init__(
            self,
            atomic_number,
            density_mg_per_mm3,
            thickness_mm
            ):
        super(SpectralAttenuator, self).__init__()

        """
        Initialize a SpectralAttenuator object.

        This class is a model of a general spectral attenuator. The list of atomic numbers is constant for all batches. For each batch element, there is one density and filter thickness. All batches are processed independently and in parallel. This class is a subclass of nn.Module and can be used in a neural network. The behavior of this class as an operator is defined by the forward() function.

        Args:
            atomic_number (torch.Tensor): The atomic numbers of the elements in the attenuator. Shape: [nElement]. Type: torch.long. Description: The list of filter atomic numbers is constant for all batches.
            density_mg_per_mm3 (torch.Tensor): The density of the filter in g/mm^3. Shape: [nBatch, nElement]. Type: torch.float32. Description: For each batch element, there is one filter density.
            thickness_mm (torch.Tensor): The thickness of the filter in mm. Shape: [nBatch, nElement]. Type: torch.float32. Description: For each batch element, there is one filter thickness.

        Example:
            >>> import torch
            >>> import miel.spectral
            >>> device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            >>> atomic_number = torch.tensor([13], dtype=torch.long).reshape([1,1]).to(device)
            >>> density_mg_per_mm3 = torch.tensor([2.7], dtype=torch.float32).reshape([1,1]).to(device)
            >>> thickness_mm = torch.tensor([1.0], dtype=torch.float32).reshape([1,1]).to(device)
            >>> al_filter = miel.spectral.SpectralAttenuator(atomic_number, density_mg_per_mm3, thickness_mm)
            >>> transmission_probability_spectrum = al_filter.compute_transmission_probability_spectrum()
            >>> print(transmission_probability_spectrum.shape)
            torch.Size([1, 150])
        """

        # make sure everything is a torch tensor
        assert isinstance(atomic_number, torch.Tensor), "atomic_number must be a torch.Tensor."
        assert isinstance(density_mg_per_mm3, torch.Tensor), "density_mg_per_mm3 must be a torch.Tensor."
        assert isinstance(thickness_mm, torch.Tensor), "thickness_mm must be a torch.Tensor."

        # make sure everything is the right type
        assert atomic_number.dtype == torch.long, "atomic_number must be a torch.long (long integer) tensor."
        assert density_mg_per_mm3.dtype == torch.float32, "density_mg_per_mm3 must be a torch.float32 tensor."
        assert thickness_mm.dtype == torch.float32, "thickness_mm must be a torch.float32 tensor."

        # handle the shapes
        if atomic_number.shape == torch.Size([]):
            atomic_number = atomic_number.reshape(1)
        if density_mg_per_mm3.shape == torch.Size([]):
            density_mg_per_mm3 = density_mg_per_mm3.reshape(1)
        self.nElement = atomic_number.shape[0]
        self.nBatch = density_mg_per_mm3.shape[0]
        assert atomic_number.shape == torch.Size([self.nElement]), "atomic_number must be a [nElement] tensor."
        assert density_mg_per_mm3.shape == torch.Size([self.nBatch, self.nElement]), "density_mg_per_mm3 must be a [nBatch, nElement] tensor."
        assert thickness_mm.shape == torch.Size([self.nBatch, self.nElement]), "thickness_mm must be a [nBatch, nElement] tensor."
        
        self.atomic_number = atomic_number
        self.density_mg_per_mm3 = density_mg_per_mm3
        self.thickness_mm = thickness_mm

        return

    def compute_transmission_probability_spectrum(self):
        """
        Compute the transmission probability spectrum of the spetral attenuator.

        This function computes the transmission probability spectrum of the filter using Beer's Law. 
        
        The transmission probability is the energy-dependent probability that an incident x-ray photon will be transmitted.

        Returns:
            transmission_probability_spectrum (torch.Tensor): The transmission probability spectrum of the filter. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The probability that an incident x-ray photon will be transmitted through the filter for each energy bin.
        """

        nEnergy = massAttenuationSpectra.shape[0]

        transmission_probability_spectrum = torch.ones(1,nEnergy).to(device)
        for iElement in range(self.nElement):
            mass_attenuation_spectrum = massAttenuationSpectra[:,self.atomic_number[iElement]].reshape(1,nEnergy)
            transmission_probability_spectrum = transmission_probability_spectrum*torch.exp(-mass_attenuation_spectrum*self.thickness_mm[:,iElement:iElement+1]*self.density_mg_per_mm3[:,iElement:iElement+1])
        return transmission_probability_spectrum

    def forward(self, incident_xray_spectrum):
        """
        Compute the filter transmission x-ray spectrum.

        This function applies the filter to the input x-ray spectrum to compute the expected number of transmitted x-ray photons for each energy bin.

        Args:
            incident_xray_spectrum (torch.Tensor): The expected number of x-ray photons of the incident x-rays for each energy bin. Shape: [nBatch, nEnergy]. Type: torch.float32. Description: The expected number of x-ray photons of the incident x-rays for each energy bin.

        Returns:
            filter_transmission_xray_spectrum (torch.Tensor): The expected number of x-ray photons of the transmitted x-rays for each energy bin. Shape: [nBatch, nEnergy]. Type: torch.float32. Description: The expected number of x-ray photons of the transmitted x-rays for each energy bin.
        """
            
        # compute the transmission probability spectrum of the filters
        filter_transmission_probability_spectrum = self.compute_filter_transmission_probability_spectrum()

        # apply the filters to the input data
        filter_transmission_xray_spectrum = incident_xray_spectrum * filter_transmission_probability_spectrum

        return filter_transmission_xray_spectrum


class SpectralSource(nn.Module):
    def __init__(
            self
            ):
        """
        This is a base class for spectral x-ray sources. It is not meant to be used directly.

        Any subclass of SpectralSource must implement the following methods:
        
        compute_transmission_spectrum()

        This method has no arguments and returns the transmission spectrum of the x-ray source.

        It will always be interpretted as the number of x-ray photons per energy bin per exposure.

        The shape of the returned tensor must be  [nBatch, nSourceChannel, nEnergy].
        

        Args:
            expected_photons_per_exposure (torch.Tensor): The expected number of photons per exposure from the xray source. Shape: [nBatch]. Type: torch.float32. Description: The expected number of photons per exposure from the xray source.
            peak_voltage_kv (torch.Tensor): The peak voltage of the xray source. Shape: [nBatch]. Type: torch.float32. Description: The peak voltage of the xray source.

        Example:
            N/A. Do not use this abstract class directly.
        """

        super().__init__()

        return
    
    def compute_transmission_spectrum(self):
        """
        Compute the transmission spectrum of the x-ray source.

        This is an abstract method that must be implemented by any subclass of SpectralSource.

        Returns:
            source_transmission_spectrum (torch.Tensor): The transmission spectrum of the x-ray source. Shape: [nBatch, nSourceChannel, nEnergy]. Type: torch.float32. Description: The transmission spectrum of the x-ray source.
        """

        raise NotImplementedError("This method must be implemented by any subclass of SpectralSource.")

    def forward(self, attenuator_transmission_probability_spectrum):
        """
        Compute the x-ray spectrum of the x-ray source.

        This function computes the x-ray spectrum of the x-ray source.

        Args:
            attenuator_transmission_probability_spectrum (torch.Tensor): The transmission probability spectrum of the attenuator. Shape: [nBatch, nEnergy]. Type: torch.float32. Description: The transmission probability spectrum of the attenuator.

        Returns:
            attenuator_transmission_spectrum (torch.Tensor): The number of x-ray photons transmitted through the attenuator for each energy bin. Shape: [nBatch, nEnergy]. Type: torch.float32. Description: The number of x-ray photons transmitted through the attenuator for each energy bin.
        """
        # compute the transmission spectrum of the x-ray source
        source_transmission_spectrum = self.compute_transmission_spectrum()

        assert isinstance(source_transmission_spectrum, torch.Tensor), "The source transmission spectrum must be a torch.Tensor."
        assert source_transmission_spectrum.dtype == torch.float32, "The source transmission spectrum must be of type torch.float32."
        assert source_transmission_spectrum.dim() == 3, "The source transmission spectrum must have 3 dimensions. [nBatch, nSourceChannel, nEnergy]"
        assert source_transmission_spectrum.shape[0] == attenuator_transmission_probability_spectrum.shape[0], "The source transmission spectrum must have the same number of batches as the attenuator transmission probability spectrum."
        assert source_transmission_spectrum.shape[2] == attenuator_transmission_probability_spectrum.shape[1], "The source transmission spectrum must have the same number of energy bins as the attenuator transmission probability spectrum."

        # reshape attenuator_transmission_probability_spectrum to [nBatch, 1, nEnergy]
        attenuator_transmission_probability_spectrum = torch.unsqueeze(attenuator_transmission_probability_spectrum, dim=1)

        # compute the spectrum of the x-rays transmitted through the attenuator
        attenuator_transmission_spectrum = source_transmission_spectrum * attenuator_transmission_probability_spectrum

        return attenuator_transmission_spectrum
    


class SpectralSource_TASMICS(SpectralSource):
    def __init__(
            self,
            expected_photons_per_exposure,
            peak_voltage_kv,
            ):
        """

        Initialize the spectral x-ray source model.

        @article{hernandez2014tungsten,
            title={Tungsten anode spectral model using interpolating cubic splines: unfiltered x-ray spectra from 20 kV to 640 kV},
            author={Hernandez, Andrew M and Boone, John M},
            journal={Medical physics},
            volume={41},
            number={4},
            pages={042101},
            year={2014},
            publisher={Wiley Online Library}

        Args:
            expected_photons_per_exposure (torch.Tensor): The expected number of photons per exposure from the xray source. Shape: [nBatch]. Type: torch.float32. Description: The expected number of photons per exposure from the xray source.
            peak_voltage_kv (torch.Tensor): The peak voltage of the xray source. Shape: [nBatch]. Type: torch.float32. Description: The peak voltage of the xray source.

        Example:
            >>> import torch
            >>> import miel
            >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            >>> nBatch = 11
            >>> nEnergy = 150
            >>> peak_voltage_kv = torch.ones([nBatch]).to(device)*120.0
            >>> expected_photons_per_exposure = torch.ones([nBatch]).to(device)*1e8
            >>> source = miel.spectral.SpectralSource(expected_photons_per_exposure, peak_voltage_kv)
            >>> source_transmission_xray_spectrum = source.compute_source_transmission_xray_spectrum()
        """

        super(SpectralSource_TASMICS, self).__init__()

        # there is one source channel
        self.nSourceChannel = 1

        # make sure everything is a torch tensor
        assert isinstance(expected_photons_per_exposure, torch.Tensor), "expected_photons_per_exposure must be a torch.Tensor of shape (nBatch,)."
        assert isinstance(peak_voltage_kv, torch.Tensor), "peak_voltage_kv must be a torch.Tensor of shape (nBatch,)."

        # define some critical shapes we will need
        nBatch = expected_photons_per_exposure.shape[0]

        # make sure everything is the right shape
        assert expected_photons_per_exposure.shape == torch.Size([nBatch]), "expected_photons_per_exposure must be a torch.Tensor of shape (nBatch,)."
        assert peak_voltage_kv.shape == torch.Size([nBatch]), "peak_voltage_kv must be a torch.Tensor of shape (nBatch,)."
        # make sure everything is the right type
        assert expected_photons_per_exposure.dtype == torch.float32, "expected_photons_per_exposure must be a torch.Tensor of type torch.float32."
        assert peak_voltage_kv.dtype == torch.float32, "peak_voltage_kv must be a torch.Tensor of type torch.float32."

        # define some useful constants
        self.nBatch = nBatch
        self.nEnergy = photon_kev.shape[0]

        # define the attributes
        self.expected_photons_per_exposure = expected_photons_per_exposure
        self.peak_voltage_kv = peak_voltage_kv

        return

    def compute_transmission_spectrum(self):
        
        """
        This function computes the source transmission xray spectrum.

        Args:
            None

        Returns:
            source_transmission_xray_spectrum (torch.Tensor): The source transmission xray spectrum. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The source transmission xray spectrum.
        """

        # define the source spectrum using the TASMICS model
        source_spectrum = get_TASMICS_Spectrum(self.peak_voltage_kv).to(torch.float32).to(device)
        
        # verify source_spectrum should be a tensor with shape [nBatch x nEnergy] 
        assert torch.is_tensor(source_spectrum), "source_spectrum must be a torch.Tensor."
        assert source_spectrum.shape == (self.nBatch, self.nEnergy), "source_spectrum must be shaped [nBatch x nEnergy]."

        # scale the source spectrum to the expected photons per exposure
        source_spectrum *= self.expected_photons_per_exposure.reshape([self.nBatch, 1])/torch.sum(source_spectrum)

        # assert the source spectrum is a torch tensor
        assert torch.is_tensor(source_spectrum), "Something went wrong, source_spectrum must be a torch.Tensor."
        assert source_spectrum.shape == (self.nBatch, self.nEnergy), "'Something went wrong, the source_spectrum must be shaped [nBatch x nEnergy]."

        # reshape the source spectrum to [nBatch x nSourceChannel x nEnergy]
        source_spectrum = source_spectrum.reshape([self.nBatch, self.nSourceChannel, self.nEnergy])

        return source_spectrum

    def forward(self, attenuator_transmission_probability_spectrum):
        """

        This function takes as input the attenuator transmission probability spectrum and returns what the transmitted x-ray spectrum would be if this source is transmitted through the attenuator.
        
        Args:
            attenuator_transmission_probability_spectrum (torch.Tensor): The attenuator transmission probability spectrum. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The attenuator transmission probability spectrum.

        Returns:
            attenuator_transmission_xray_spectrum (torch.Tensor): The attenuator transmission xray spectrum. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The attenuator transmission xray spectrum.

        """

        # compute the expected number of photons generated by the source
        source_transmission_xray_spectrum = self.compute_transmission_spectrum()

        # compute the expected number of photons transmitted through the attenuator
        attenuator_transmission_xray_spectrum = source_transmission_xray_spectrum.reshape([self.nBatch, self.nEnergy, 1]) * attenuator_transmission_probability_spectrum

        return attenuator_transmission_xray_spectrum





class SpectralSource_TASMICS_Filter(SpectralSource_TASMICS):
    def __init__(
            self,
            expected_photons_per_exposure,
            peak_voltage_kv,
            filter_atomic_number,
            filter_density_mg_per_mm3,
            filter_thickness_mm,
            ):
        """
        Initialize the spectral x-ray source model.

        This function initializes the spectral x-ray source model.

        Args:
            expected_photons_per_exposure (torch.Tensor): The expected number of photons per exposure from the xray source. Shape: [nBatch]. Type: torch.float32. Description: The expected number of photons per exposure from the xray source.
            peak_voltage_kv (torch.Tensor): The peak voltage of the xray source. Shape: [nBatch]. Type: torch.float32. Description: The peak voltage of the xray source.
            length_aluminium_filter_mm (torch.Tensor): The length of the aluminium filter in the xray source. Shape: [nBatch]. Type: torch.float32. Description: The length of the aluminium filter in the xray source.

        Example:
            >>> import torch
            >>> import miel
            >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            >>> nBatch = 11
            >>> peak_voltage_kv = torch.ones([nBatch]).to(device)*120.0
            >>> expected_photons_per_exposure = torch.ones([nBatch]).to(device)*1e8
            >>> filter_atomic_number = torch.ones([1], dtype=torch.long).to(device)*13
            >>> filter_density_mg_per_mm3 = torch.ones([nBatch,1]).to(device)*2.7
            >>> filter_thickness_mm = torch.linspace(0.0, 5.0, nBatch).to(device).reshape([nBatch,1])
            >>> source = miel.spectral.SpectralSource_TASMICS_Filter(expected_photons_per_exposure, peak_voltage_kv, filter_atomic_number, filter_density_mg_per_mm3, filter_thickness_mm)
            >>> source_transmission_xray_spectrum = source.forward(attenuator_transmission_probability_spectrum)
            >>> average_kev_of_spectrum = torch.sum(source_transmission_xray_spectrum*miel.spectral.photon_kev, dim=1)/torch.sum(source_transmission_xray_spectrum, dim=1)
        """

        # parent class is SpectralSource
        # parent class of SpectralSource is nn.Module
        # run the nn.Module.__init__ function
        nn.Module.__init__(self)

        # there is only one source channel
        self.nSourceChannel = 1

        # make sure everything is a torch tensor
        assert isinstance(expected_photons_per_exposure, torch.Tensor), "expected_photons_per_exposure must be a torch.Tensor of shape (nBatch,)."
        assert isinstance(peak_voltage_kv, torch.Tensor), "peak_voltage_kv must be a torch.Tensor of shape (nBatch,)."
        assert isinstance(filter_atomic_number, torch.Tensor), "filter_atomic_number must be a torch.Tensor of shape (nBatch, 1)."
        assert isinstance(filter_density_mg_per_mm3, torch.Tensor), "filter_density_mg_per_mm3 must be a torch.Tensor of shape (nBatch, 1)."
        assert isinstance(filter_thickness_mm, torch.Tensor), "filter_thickness_mm must be a torch.Tensor of shape (nBatch, 1)."


        # define some critical shapes we will need
        nBatch = expected_photons_per_exposure.shape[0]
        nElement = filter_atomic_number.shape[0]

        # make sure everything is the right shape
        assert expected_photons_per_exposure.shape == torch.Size([nBatch]), "expected_photons_per_exposure must be a torch.Tensor of shape (nBatch,)."
        assert peak_voltage_kv.shape == torch.Size([nBatch]), "peak_voltage_kv must be a torch.Tensor of shape (nBatch,)."
        assert filter_atomic_number.shape == torch.Size([nElement]), "filter_atomic_number must be a torch.Tensor of shape (nBatch, 1)."
        assert filter_density_mg_per_mm3.shape == torch.Size([nBatch, nElement]), "filter_density_mg_per_mm3 must be a torch.Tensor of shape (nBatch, 1)."
        assert filter_thickness_mm.shape == torch.Size([nBatch, nElement]), "filter_thickness_mm must be a torch.Tensor of shape (nBatch, 1)."

        # make sure everything is the right type
        assert expected_photons_per_exposure.dtype == torch.float32, "expected_photons_per_exposure must be a torch.Tensor of type torch.float32."
        assert peak_voltage_kv.dtype == torch.float32, "peak_voltage_kv must be a torch.Tensor of type torch.float32."
        assert filter_atomic_number.dtype == torch.long, "filter_atomic_number must be a torch.Tensor of type torch.long."
        assert filter_density_mg_per_mm3.dtype == torch.float32, "filter_density_mg_per_mm3 must be a torch.Tensor of type torch.float32."
        assert filter_thickness_mm.dtype == torch.float32, "filter_thickness_mm must be a torch.Tensor of type torch.float32."

        # define some useful constants
        self.nBatch = nBatch
        self.nEnergy = photon_kev.shape[0]

        # define the attributes
        # self.expected_photons_per_exposure = expected_photons_per_exposure
        # self.peak_voltage_kv = peak_voltage_kv

        # run the super class __init__ function
        super(SpectralSource_TASMICS_Filter, self).__init__(expected_photons_per_exposure, peak_voltage_kv)
        
        self.filter_atomic_number = filter_atomic_number
        self.filter_density_mg_per_mm3 = filter_density_mg_per_mm3
        self.filter_thickness_mm = filter_thickness_mm


        # define the aluminium filter
        nBatch = self.nBatch

        self.filter = SpectralAttenuator(
                                    filter_atomic_number,
                                    filter_density_mg_per_mm3,
                                    filter_thickness_mm,
                                    ).to(device)
        return

    def compute_transmission_spectrum(self):
        """
        This function computes the source transmission xray spectrum.

        Args:
            None

        Returns:
            source_transmission_spectrum (torch.Tensor): The source transmission xray spectrum. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The source transmission xray spectrum.
        
        Example:
            >>> from miel.spectral import SpectralSource
            >>> expected_photons_per_exposure = torch.tensor([1000.0, 2000.0, 3000.0])
            >>> peak_voltage_kv = torch.tensor([80.0, 100.0, 120.0])
            >>> length_aluminium_filter_mm = torch.tensor([0.0, 0.5, 1.0])
            >>> source = miel.spectral.SpectralSource(expected_photons_per_exposure, peak_voltage_kv, length_aluminium_filter_mm)
            >>> source_transmission_xray_spectrum = source.compute_source_transmission_xray_spectrum()
        """

        # # define the source spectrum using the TASMICS model
        # source_spectrum = get_TASMICS_Spectrum(self.peak_voltage_kv).to(torch.float32).to(device)
        
        # # verify source_spectrum should be a tensor with shape [nBatch x nEnergy] 
        # assert torch.is_tensor(source_spectrum), "source_spectrum must be a torch.Tensor."
        # assert source_spectrum.shape == (self.nBatch, self.nEnergy), "source_spectrum must be shaped [nBatch x nEnergy]."

        # run the super class function
        source_spectrum = super(SpectralSource_TASMICS_Filter, self).compute_transmission_spectrum()

        # reshape it to [nBatch x nEnergy]
        source_spectrum = source_spectrum.reshape([self.nBatch, self.nEnergy])

        # attenuate the source spectrum by the aluminium filter
        filter_transmission_probability_spectrum = self.filter.compute_transmission_probability_spectrum()
        assert filter_transmission_probability_spectrum.shape == (self.nBatch, self.nEnergy), "Something went wrong, filter_transmission_probability_spectrum must be shaped [nBatch x nEnergy]."
        source_spectrum *= filter_transmission_probability_spectrum

        # # scale the source spectrum to the expected photons per exposure
        # source_spectrum *= self.expected_photons_per_exposure.reshape([self.nBatch, 1])/torch.sum(source_spectrum)

        # # assert the source spectrum is a torch tensor
        # assert torch.is_tensor(source_spectrum), "Something went wrong, source_spectrum must be a torch.Tensor."
        # assert source_spectrum.shape == (self.nBatch, self.nEnergy), "'Something went wrong, the source_spectrum must be shaped [nBatch x nEnergy]."

        # reshape the source spectrum to [nBatch x nSourceChannel x nEnergy]
        source_spectrum = source_spectrum.reshape([self.nBatch, 1, self.nEnergy])

        return source_spectrum

    def forward(self, attenuator_transmission_probability_spectrum):
        """
        
        This function computes the expected number of photons transmitted through the attenuator.

        Args:
            attenuator_transmission_probability_spectrum (torch.Tensor): The attenuator transmission probability spectrum. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The attenuator transmission probability spectrum.

        Returns:
            attenuator_transmission_xray_spectrum (torch.Tensor): The attenuator transmission xray spectrum. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The attenuator transmission xray spectrum.
            
        """

        # compute the expected number of photons generated by the source
        source_transmission_xray_spectrum = self.compute_transmission_spectrum()

        # compute the expected number of photons transmitted through the attenuator
        attenuator_transmission_xray_spectrum = source_transmission_xray_spectrum.reshape([self.nBatch, self.nEnergy, 1]) * attenuator_transmission_probability_spectrum

        return attenuator_transmission_xray_spectrum




class SpectralDetctor(nn.Module):
    def __init__(self):
        """
        This is an abstract class for a spectral detector.
        
        it should implement the following methods:
            compute_interaction_probability_spectrum()
            compute_detector_coversion_spectrum()
            compute_gain()
        """
        super(SpectralDetctor, self).__init__()
        self.nDetectorChannel = 1
        return
    def compute_interaction_probability_spectrum(self):
        raise NotImplementedError
    def compute_detector_coversion_spectrum(self):
        raise NotImplementedError
    def compute_gain(self):
        raise NotImplementedError
    def forward(self, incident_spectrum):
        """
        This function computes the expected number of photons detected by the detector.

        Args:
            incident_spectrum (torch.Tensor): The incident spectrum. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The incident spectrum.

        Returns:
            detector_measurements (torch.Tensor): The detector measurements. Shape: [nBatch]. Type: torch.float32. Description: The detected spectrum.
        """
        # compute the interaction probability spectrum
        interaction_probability_spectrum = self.compute_interaction_probability_spectrum()

        assert torch.is_tensor(interaction_probability_spectrum), "interaction_probability_spectrum must be a torch.Tensor."
        assert len(interaction_probability_spectrum.shape) == 3, "interaction_probability_spectrum must be shaped [nBatch x nDetectorChannel x nEnergy ]."
        nBatch, nDetectorChannel, nEnergy = interaction_probability_spectrum.shape

        # compute the conversion spectrum
        conversion_spectrum = self.compute_detector_coversion_spectrum()

        assert torch.is_tensor(conversion_spectrum), "conversion_spectrum must be a torch.Tensor."
        assert len(conversion_spectrum.shape) == 3, "conversion_spectrum must be shaped [nBatch x nDetectorChannel x nEnergy ]."
        assert conversion_spectrum.shape == (nBatch, nDetectorChannel, nEnergy), "conversion_spectrum must be shaped [nBatch x nDetectorChannel x nEnergy ]."

        # compute the gain
        gain = self.compute_gain()

        assert torch.is_tensor(gain), "gain must be a torch.Tensor."
        assert len(gain.shape) == 2, "gain must be shaped [nBatch x nDetectorChannel]."
        assert gain.shape == (nBatch, nDetectorChannel), "gain must be shaped [nBatch x nDetectorChannel]."

        # compute the detected spectrum
        detector_measurements = torch.sum(incident_spectrum.reshape([nBatch, 1, nEnergy]) * interaction_probability_spectrum * conversion_spectrum, dim=2) * gain
    
        return detector_measurements





class FlatPanelDetector(nn.Module):
    def __init__(
            self,
            detector_thickness_CsI_mm
            ):
        """
        This class is a model of a flat panel x-ray detector with a CsI scintillator.

        Args:
            detector_thickness_CsI_mm (torch.Tensor): The thickness of the CsI scintillator. Shape: [nBatch]. Type: torch.float32. Description: The thickness of the CsI scintillator.

        Example:
            >>> import torch
            >>> import miel
            >>> nBatch = 3
            >>> detector_thickness_CsI_mm = torch.tensor([0.0, 0.5, 1.0]).reshape([nBatch])
            >>> detector = miel.spectral.FlatPanelDetector(detector_thickness_CsI_mm)
            >>> detector_transmission_probability_spectrum = detector.compute_detector_transmission_probability_spectrum()
            >>> detector_interaction_probability_spectrum = detector.compute_interaction_probability_spectrum()
        """

        super(FlatPanelDetector, self).__init__()

        # there is one detector channel
        self.nDetectorChannel = 1

        # make sure everything is a torch tensor
        assert isinstance(detector_thickness_CsI_mm, torch.Tensor), "detector_thickness_CsI_mm must be a torch.Tensor."
        assert detector_thickness_CsI_mm.dtype == torch.float32, "detector_thickness_CsI_mm must be a torch.float32 tensor."
        self.nBatch = detector_thickness_CsI_mm.shape[0]
        assert detector_thickness_CsI_mm.shape == torch.Size([self.nBatch]), "detector_thickness_CsI_mm must be shaped [nBatch]."

        self.detector_thickness_CsI_mm = detector_thickness_CsI_mm

         # define the detector interaction spectrum (probability of interaction with detector)
        rho_CsI = torch.tensor(4.51).to(torch.float32).repeat([1,1]).to(device) # mg / mm^2 density of CsI
        rho_Cs_in_CsI = 0.5*rho_CsI # mg/mm3 density of Cs in CsI
        rho_I_in_CsI = 0.5*rho_CsI # mg/mm3 density of I in CsI
        atomic_number_Cs = torch.tensor(55).to(torch.long).repeat([1,1]).to(device) # atomic number of Cs
        atomic_number_I = torch.tensor(53).to(torch.long).repeat([1,1]).to(device) # atomic number of I

        # the probability of transmission through the first layer using Beer's law
        atomic_number = torch.tensor([atomic_number_I, atomic_number_Cs]).to(torch.long).reshape([2]).to(device) # atomic number of I and Cs
        density_mg_per_mm3 = torch.tensor([rho_I_in_CsI, rho_Cs_in_CsI]).to(torch.float32).reshape([1,2]).repeat([self.nBatch,1]).to(device) # density of I and Cs in CsI
        thickness_mm = self.detector_thickness_CsI_mm.clone().to(torch.float32).reshape([self.nBatch,1]).repeat([1,2]).to(device) # thickness of CsI
        
        self.scintillator = SpectralAttenuator(
                                atomic_number,
                                density_mg_per_mm3,
                                thickness_mm
                                )

        self.set_gain(torch.ones([self.nBatch, self.nDetectorChannel]).to(torch.float32))

        return

    def compute_detector_transmission_probability_spectrum(self):
        """
        This function computes the transmission probability spectrum of the filter.

        Returns:
            detector_transmission_probability_spectrum (torch.Tensor): The detector transmission probability spectrum. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The probability that an incident x-ray photon will be transmitted through the detector for each energy bin. This is the output of Beer's Law: exp(-mu*l) for the CsI detector.
        """
        # compute the transmission probability spectrum of the filter
        detector_transmission_probability_spectrum = self.scintillator.compute_transmission_probability_spectrum()

        return detector_transmission_probability_spectrum

    def compute_interaction_probability_spectrum(self):
        """
        This function computes the interaction probability spectrum of the detector.

        Returns:
            detector_interaction_probability_spectrum (torch.Tensor): The detector interaction probability spectrum. Shape: [nBatch x nEnergy]. Type: torch.float32. Description: The probability that an incident x-ray photon will interact with the detector for each energy bin. This is the output of [1 - exp(-mu*l)] for the filter.
        """

        # compute the transmission probability spectrum
        detector_transmission_probability_spectrum = self.compute_detector_transmission_probability_spectrum()

        # compute the interaction probability spectrum
        detector_interaction_probability_spectrum = 1.0 - detector_transmission_probability_spectrum

        return detector_interaction_probability_spectrum

    def compute_conversion_spectrum(self):
        """
        This function computes the conversion spectrum of the detector.

        Returns:
            detector_conversion_spectrum (torch.Tensor): The detector conversion gain spectrum. Shape: [nBatch x 1 x nEnergy]. Type: torch.float32. Description: The conversion gain of the detector for each energy bin.
        """

        # compute the conversion gain spectrum
        detector_conversion_gain_spectrum = (photon_kev/torch.sum(photon_kev)).reshape(1,nEnergy).repeat([self.nBatch,1]).to(device)
        
        # reshape the conversion gain spectrum
        detector_conversion_gain_spectrum = detector_conversion_gain_spectrum.reshape([self.nBatch,1,nEnergy])

        return detector_conversion_gain_spectrum
    
    def compute_gain(self):
        """
        This function computes the gain spectrum of the detector.

        Returns:
            gain (torch.Tensor): The detector gain spectrum. Shape: [nBatch x 1 ]. Type: torch.float32. Description: The gain of the detector.
        """

        assert self.gain.shape == torch.Size([self.nBatch,1]), "gain must be shaped [nBatch,1]."
        
        return self.gain

    def set_gain(self, gain):
        """
        This function sets the gain of the detector.

        Args:
            gain (torch.Tensor): The detector gain spectrum. Shape: [nBatch x 1 x nEnergy]. Type: torch.float32. Description: The gain of the detector for each energy bin.
        """
        assert type(gain) == torch.Tensor, "gain must be a torch.Tensor."
        assert gain.dtype == torch.float32, "gain must be a torch.float32."
        assert gain.shape == torch.Size([self.nBatch,1]), "gain must be shaped [nBatch]."
        self.gain = gain

        return



class DualLayerFlatPanelDetector(nn.Module):
    def __init__(
            self,
            layer1_CsI_thickness_mm,
            filter_atomic_number,
            filter_density_mg_per_mm3,
            filter_thickness_mm,
            layer2_CsI_thickness_mm,
            gain=None
            ):
        """
        This class implements a dual layer flat panel detector.

        Args:
            layer1_CsI_thickness_mm (torch.Tensor): The thickness of the first CsI layer in mm. Shape: [nBatch]. Type: torch.float32. Description: The thickness of the first CsI layer in mm.
            filter_atomic_number (torch.Tensor): The atomic number of the filter. Shape: [nBatch]. Type: torch.float32. Description: The atomic number of the filter.
            filter_density_mg_per_mm3 (torch.Tensor): The density of the filter in mg/mm^3. Shape: [nBatch]. Type: torch.float32. Description: The density of the filter in mg/mm^3.
            filter_thickness_mm (torch.Tensor): The thickness of the filter in mm. Shape: [nBatch]. Type: torch.float32. Description: The thickness of the filter in mm.
            layer2_CsI_thickness_mm (torch.Tensor): The thickness of the second CsI layer in mm. Shape: [nBatch]. Type: torch.float32. Description: The thickness of the second CsI layer in mm.
        
        Example:
            >>> import torch
            >>> import miel
            >>> nBatch = 3
            >>> layer1_CsI_thickness_mm = torch.tensor([0.0, 0.5, 1.0]).reshape([nBatch]).to(device)
            >>> filter_atomic_number = torch.tensor([29]).reshape([1]).to(device)
            >>> filter_density_mg_per_mm3 = torch.tensor([0.0, 0.5, 1.0]).reshape([nBatch, 1]).to(device)
            >>> filter_thickness_mm = torch.tensor([0.0, 0.5, 1.0]).reshape([nBatch, 1]).to(device)
            >>> layer2_CsI_thickness_mm = torch.tensor([0.0, 0.5, 1.0]).reshape([nBatch]).to(device)
            >>> detector = miel.spectral.DualLayerFlatPanelDetector(layer1_CsI_thickness_mm, filter_atomic_number, filter_density_mg_per_mm3, filter_thickness_mm, layer2_CsI_thickness_mm)
            >>> layer1_transmission_probability_spectrum = detector.compute_layer1_transmission_probability_spectrum()
            >>> layer2_transmission_probability_spectrum = detector.compute_layer2_transmission_probability_spectrum()
            >>> filter_transmission_probability_spectrum = detector.compute_filter_transmission_probability_spectrum()
            >>> detector_interaction_probability_spectrum = detector.compute_interaction_probability_spectrum()
            >>> detector_conversion_gain_spectrum = detector.compute_detector_conversion_gain_spectrum()
        """

        super(DualLayerFlatPanelDetector, self).__init__()

        # two detector channels
        self.nDetectorChannel = 2

        # make sure everything is a tensor
        assert isinstance(layer1_CsI_thickness_mm, torch.Tensor), 'layer1_CsI_thickness_mm must be a torch.Tensor'
        assert isinstance(filter_atomic_number, torch.Tensor), 'filter_atomic_number must be a torch.Tensor'
        assert isinstance(filter_density_mg_per_mm3, torch.Tensor), 'filter_density_mg_per_mm3 must be a torch.Tensor'
        assert isinstance(filter_thickness_mm, torch.Tensor), 'filter_thickness_mm must be a torch.Tensor'
        assert isinstance(layer2_CsI_thickness_mm, torch.Tensor), 'layer2_CsI_thickness_mm must be a torch.Tensor'

        # make sure everything is the right type
        assert layer1_CsI_thickness_mm.dtype == torch.float32, 'layer1_CsI_thickness_mm must be a torch.float32'
        assert filter_atomic_number.dtype == torch.long, 'filter_atomic_number must be a torch.long'
        assert filter_density_mg_per_mm3.dtype == torch.float32, 'filter_density_mg_per_mm3 must be a torch.float32'
        assert filter_thickness_mm.dtype == torch.float32, 'filter_thickness_mm must be a torch.float32'
        assert layer2_CsI_thickness_mm.dtype == torch.float32, 'layer2_CsI_thickness_mm must be a torch.float32'

        # make sure everything has the right number of dimensions
        assert layer1_CsI_thickness_mm.dim() == 1, 'layer1_CsI_thickness_mm must be a 1D tensor of shape [nBatch]'
        assert filter_atomic_number.dim() == 1, 'filter_atomic_number must be a 1D tensor of shape [nElement]'
        assert filter_density_mg_per_mm3.dim() == 2, 'filter_density_mg_per_mm3 must be a 2D tensor of shape [nBatch, nElement]'
        assert filter_thickness_mm.dim() == 2, 'filter_thickness_mm must be a 2D tensor of shape [nBatch, nElement]'
        assert layer2_CsI_thickness_mm.dim() == 1, 'layer2_CsI_thickness_mm must be a 1D tensor of shape [nBatch]'

        # define some constants
        self.nBatch = layer1_CsI_thickness_mm.shape[0]
        self.nFilterElement = filter_atomic_number.shape[0]
        self.nEnergy = photon_kev.shape[0]

        # make sure everything has the right shape
        assert filter_density_mg_per_mm3.shape == torch.Size([self.nBatch, self.nFilterElement]), 'filter_density_mg_per_mm3 must be a 2D tensor of shape [nBatch, nElement]'
        assert filter_thickness_mm.shape == torch.Size([self.nBatch, self.nFilterElement]), 'filter_thickness_mm must be a 2D tensor of shape [nBatch, nElement]'
        assert layer1_CsI_thickness_mm.shape == torch.Size([self.nBatch]), 'layer1_CsI_thickness_mm must be a 1D tensor of shape [nBatch]'
        assert layer2_CsI_thickness_mm.shape == torch.Size([self.nBatch]), 'layer2_CsI_thickness_mm must be a 1D tensor of shape [nBatch]'

        # save the attributes
        self.layer1_CsI_thickness_mm = layer1_CsI_thickness_mm
        self.layer2_CsI_thickness_mm = layer2_CsI_thickness_mm
        self.filter_atomic_number = filter_atomic_number
        self.filter_density_mg_per_mm3 = filter_density_mg_per_mm3
        self.filter_thickness_mm = filter_thickness_mm
        
        # make a filter object for the interstitial filter
        self.filter = SpectralAttenuator(
                            filter_atomic_number, 
                            filter_density_mg_per_mm3, 
                            filter_thickness_mm
                            )

        # make a FlatPanelDetector for the first layer
        self.layer1_FPD = FlatPanelDetector(
                            layer1_CsI_thickness_mm,
                            )
        
        # make a FlatPanelDetector for the second layer
        self.layer2_FPD = FlatPanelDetector(
                            layer2_CsI_thickness_mm,
                            )
        
        
        self.set_gain(torch.ones(self.nBatch, self.nDetectorChannel, dtype=torch.float32))

        return

    def compute_layer1_transmission_probability_spectrum(self):
        """
        This function computes the transmission probability spectrum of the first layer of CsI.

        Returns:
            layer1_transmission_probability_spectrum (torch.Tensor): The transmission probability spectrum of the first layer of CsI. Shape: [nBatch, nEnergy]. Type: torch.float32. Description: The transmission probability spectrum of the first layer of CsI. This is the probability that a photon will be transmitted through the first layer of CsI.
        """
       
        # compute the transmission probability spectrum of the first layer of CsI
        layer1_transmission_probability_spectrum = self.layer1_FPD.scintillator.compute_transmission_probability_spectrum()

        return layer1_transmission_probability_spectrum

    def compute_filter_transmission_probability_spectrum(self):
        """
        This function computes the transmission probability spectrum of the interstitial filter.

        Returns:
            filter_transmission_probability_spectrum (torch.Tensor): The transmission probability spectrum of the interstitial filter. Shape: [nBatch, nEnergy]. Type: torch.float32. Description: The transmission probability spectrum of the interstitial filter. This is the probability that a photon will be transmitted through the interstitial filter.
        """

        # compute the transmission probability spectrum of the interstitial filter
        filter_transmission_probability_spectrum = self.filter.compute_transmission_probability_spectrum()

        return filter_transmission_probability_spectrum


    def compute_layer2_transmission_probability_spectrum(self):
        """
        This function computes the transmission probability spectrum of the second layer of CsI.

        Returns:
            layer2_transmission_probability_spectrum (torch.Tensor): The transmission probability spectrum of the second layer of CsI. Shape: [nBatch, nEnergy]. Type: torch.float32. Description: The transmission probability spectrum of the second layer of CsI. This is the probability that a photon will be transmitted through the second layer of CsI.
        """
        
        # compute the transmission probability spectrum of the second layer of CsI
        layer2_transmission_probability_spectrum = self.layer1_FPD.scintillator.compute_transmission_probability_spectrum()

        return layer2_transmission_probability_spectrum

    def compute_layer1_interaction_probability_spectrum(self):
        """
        This function computes the interaction probability spectrum of the first layer of CsI.

        Returns:
            layer1_interaction_probability_spectrum (torch.Tensor): The interaction probability spectrum of the first layer of CsI. Shape: [nBatch, nEnergy]. Type: torch.float32. Description: The interaction probability spectrum of the first layer of CsI. This is the probability that a photon will interact in the first layer of CsI.
        """

        # compute the interaction probability spectrum of the first layer of CsI
        layer1_interaction_probability_spectrum = self.layer1_FPD.compute_interaction_probability_spectrum()

        return layer1_interaction_probability_spectrum

    def compute_layer2_interaction_probability_spectrum(self):
        """
        This function computes the interaction probability spectrum of the second layer of CsI.

        Returns:
            layer2_interaction_probability_spectrum (torch.Tensor): The interaction probability spectrum of the second layer of CsI. Shape: [nBatch, nEnergy]. Type: torch.float32. Description: The interaction probability spectrum of the second layer of CsI. This is the probability that a photon will interact in the second layer of CsI.
        """

        # compute the transmission probability spectrum of the first layer of CsI
        layer1_transmission_probability_spectrum = self.layer1_FPD.scintillator.compute_transmission_probability_spectrum()

        # compute the transmission probability spectrum of the interstitial filter
        filter_transmission_probability_spectrum = self.filter.compute_transmission_probability_spectrum()

        # compute the interaction probability spectrum of the second layer of CsI
        layer2_interaction_probability_spectrum = 1.0
        layer2_interaction_probability_spectrum *= layer1_transmission_probability_spectrum
        layer2_interaction_probability_spectrum *= filter_transmission_probability_spectrum
        layer2_interaction_probability_spectrum *= self.layer2_FPD.compute_interaction_probability_spectrum()

        return layer2_interaction_probability_spectrum

    def compute_interaction_probability_spectrum(self):
        """
        This function computes the interaction probability spectrum of the detector.

        Returns:
            interaction_probability_spectrum (torch.Tensor): The interaction probability spectrum of the detector. Shape: [nBatch, 2, nEnergy]. Type: torch.float32. Description: The interaction probability spectrum of the detector. This is the probability that a photon will interact in the detector.
        """

        interaction_probability_spectrum = torch.zeros(self.nBatch, self.nDetectorChannel, nEnergy, dtype=torch.float32).to(device)

        # compute the interaction probability spectrum of the first layer of CsI
        interaction_probability_spectrum[:, 0, :] = self.compute_layer1_interaction_probability_spectrum()

        # compute the interaction probability spectrum of the second layer of CsI
        interaction_probability_spectrum[:, 1, :] = self.compute_layer2_interaction_probability_spectrum()

        return interaction_probability_spectrum

    def compute_conversion_spectrum(self):
        """
        This function computes the conversion spectrum of the detector.

        Returns:
            conversion_spectrum (torch.Tensor): The conversion spectrum of the detector. Shape: [nBatch, 2, nEnergy]. Type: torch.float32. Description: The conversion spectrum of the detector. This is the probability that a photon will convert to an electron-hole pair in the detector.
        """

        conversion_spectrum = torch.zeros(self.nBatch, self.nDetectorChannel, nEnergy, dtype=torch.float32).to(device)

        # compute the conversion spectrum of the first layer of CsI
        conversion_spectrum[:, 0:1, :] = self.layer1_FPD.compute_conversion_spectrum()

        # compute the conversion spectrum of the second layer of CsI
        conversion_spectrum[:, 1:2, :] = self.layer2_FPD.compute_conversion_spectrum()

        return conversion_spectrum

    def compute_gain(self):
        """
        This function computes the gain of the detector.

        Returns:
            gain (torch.Tensor): The gain of the detector. Shape: [nBatch, 2]. Type: torch.float32. Description: The gain of the detector. 
        """
        return self.gain

    def set_gain(self, gain):
        """
        This function sets the gain of the detector.

        Args:
            gain (torch.Tensor): The gain of the detector. Shape: [nBatch, 2]. Type: torch.float32. Description: The gain of the detector. 
        """

        assert type(gain) == torch.Tensor
        assert gain.shape == (self.nBatch, self.nDetectorChannel)
        assert gain.dtype == torch.float32

        self.gain = gain
    
        




class SpectralImagingSystem(nn.Module):
    def __init__(   self,
                    source,
                    detector):

        """
        This is a base class for spectral imaging systems.

        The purpose of this class is to define S, S0, S1, and S2.

        These design matrices can be used for modeling, prediction, and reconstruction.

        Args:
            source (SpectralSource): The source of the spectral imaging system. Type: SpectralSource. Description: The source of the spectral imaging system
            detector (SpectralDetector): The detector of the spectral imaging system. Type: SpectralDetector. Description: The detector of the spectral imaging system
        """
        super().__init__()
        self.source = source
        self.detector = detector

        self.nChannel = self.source.nSourceChannel*self.detector.nDetectorChannel

    def compute_S0(self):
        """
        This method computes the source transmission spectrum.

        Returns:
           S0 (torch.Tensor): The source transmission spectrum. Shape: [nBatch x nOut_S0 x nEnergy]. Type: torch.float32. Description: The source transmission spectrum is the expected number of photons per energy bin that are transmitted through the source.
        """
        # self.source should be a SpectralSource object
        # any SpectralSource object must have a compute_transmission_spectrum method
        # and it must return a torch.Tensor of shape [nBatch x nSourceChannel x nEnergy]
        # the final matrix S0 is of shape [nBatch x nOut_S0 x nEnergy] where nOut_S0 = nSourceChannel*nEnergy. 
        # most of this will be zeros. It is a block diagonal with a block for each channel
        # in the future it would be good to have a more efficient way to implement the diagonal matrix 

        # compute the source transmission spectrum
        source_transmission_spectrum = self.source.compute_transmission_spectrum()

        assert type(source_transmission_spectrum) == torch.Tensor, "The source_transmission_spectrum method of the source must return a torch.Tensor"
        # should be float32
        assert source_transmission_spectrum.dtype == torch.float32, "The source_transmission_spectrum method of the source must return a torch.Tensor of type torch.float32"
        # should be 3D
        assert len(source_transmission_spectrum.shape) == 3, "The source_transmission_spectrum method of the source must return a torch.Tensor of shape [nBatch x nSourceChannel x nEnergy]"
        # the shape should be [nBatch x nSourceChannel x nEnergy]
        nBatch = source_transmission_spectrum.shape[0]
        nSourceChannel = source_transmission_spectrum.shape[1]
        nEnergy = source_transmission_spectrum.shape[2]

        # reshape to [nBatch*nSourceChannel x nEnergy]
        source_transmission_spectrum = source_transmission_spectrum.reshape(nBatch*nSourceChannel, nEnergy)

        # diag_embed to [nBatch*nSourceChannel x nEnergy x nEnergy]
        source_transmission_spectrum = torch.diag_embed(source_transmission_spectrum)

        # reshape to [nBatch x nSourceChannel*nEnergy x nEnergy]
        source_transmission_spectrum = source_transmission_spectrum.reshape(nBatch, nSourceChannel*nEnergy, nEnergy)

        S0 = source_transmission_spectrum

        return S0

    def compute_S1(self):
        """
        This function computes the interaction spectrum.

        The interaction spectrum represents the probability of interaction in each energy bin for each channel.

        Returns:
            torch.Tensor: The conversion spectrum of shape [nBatch x nOut_S1 x nOut_S0]. Type torch.float32. Shape [nBatch x nDetectorChannel*nEnergy x nSourceChannel*nEnergy]. Description: The conversion spectrum used to integrate energy bins and convert to measurements for each channel.
        """
        
        nSourceChannel = self.source.nSourceChannel

        # source.detector should be a SpectralDetector object
        # any SpectralDetector object must have a compute_interaction_probability_spectrum method
        # and it must return a torch.Tensor of shape [nBatch x nDetectorChannel x nEnergy]
        # the final matrix S1 is of shape [nBatch x nOut_S1 x nOut_S0] where nOut_S1 = nDetectorChannel*nEnergy and nOut_S0 = nSourceChannel*nEnergy.
        
        # compute the interaction probability spectrum
        detector_interaction_probability_spectrum = self.detector.compute_interaction_probability_spectrum()
        # torch tensor
        assert type(detector_interaction_probability_spectrum) == torch.Tensor, "The compute_interaction_probability_spectrum method of the detector must return a torch.Tensor"
        # should be float32
        assert detector_interaction_probability_spectrum.dtype == torch.float32, "The compute_interaction_probability_spectrum method of the detector must return a torch.Tensor of type torch.float32"
        # should be 3D
        assert len(detector_interaction_probability_spectrum.shape) == 3, "The compute_interaction_probability_spectrum method of the detector must return a torch.Tensor of shape [nBatch x nDetectorChannel x nEnergy]"
        # the shape should be [nBatch x nDetectorChannel x nEnergy]
        nBatch = detector_interaction_probability_spectrum.shape[0]
        nDetectorChannel = detector_interaction_probability_spectrum.shape[1]
        nEnergy = detector_interaction_probability_spectrum.shape[2]

        # repeat the third dimension nSourceChannel times
        detector_interaction_probability_spectrum = detector_interaction_probability_spectrum.repeat(1, 1, nSourceChannel)
        # at this point the shape should be [nBatch x nDetectorChannel x nSourceChannel*nEnergy]
        # reshape to [nBatch*nDetectorChannel x nSourceChannel*nEnergy]
        detector_interaction_probability_spectrum = detector_interaction_probability_spectrum.reshape(nBatch*nDetectorChannel, nSourceChannel*nEnergy)
        # use diag_embed to get [nBatch*nDetectorChannel x nSourceChannel*nEnergy x nSourceChannel*nEnergy]
        detector_interaction_probability_spectrum = torch.diag_embed(detector_interaction_probability_spectrum)
        # reshape to [nBatch x nDetectorChannel*nSourceChannel*nEnergy x nSourceChannel*nEnergy]
        detector_interaction_probability_spectrum = detector_interaction_probability_spectrum.reshape(nBatch, nDetectorChannel*nSourceChannel*nEnergy, nSourceChannel*nEnergy)

        S1 = detector_interaction_probability_spectrum

        return S1
    
    def compute_S2(self):
        """
        This function computes the conversion spectrum.

        The conversion spectrum is the spectrum used to integrate energy bins and convert to measurements for each channel.

        Returns:
            torch.Tensor: The conversion spectrum of shape [nBatch x nOut_S2 x nOut_S1]. Type torch.float32. Shape [nBatch x nDetectorChannel*nEnergy x nSourceChannel*nEnergy]. Description: The conversion spectrum used to integrate energy bins and convert to measurements for each channel.
        """
        
        # source.detector should be a SpectralDetector object
        # any SpectralDetector object must have a compute_conversion_spectrum method
        # and it must return a torch.Tensor of shape [nBatch x nDetectorChannel x nEnergy]
        # the final matrix S1 is of shape [nBatch x nOut_S2 x nOut_S1] where nOut_S2 = nDetectorChannel*nSourceChannel and nOut_S1 = nDetectorChannel*nSourceChannel*nEnergy.

        nSourceChannel = self.source.nSourceChannel
        nDetectorChannel = self.detector.nDetectorChannel

        # compute the conversion spectrum
        detector_conversion_spectrum = self.detector.compute_conversion_spectrum()
        # torch tensor
        assert type(detector_conversion_spectrum) == torch.Tensor, "The compute_conversion_spectrum method of the detector must return a torch.Tensor"
        # should be float32
        assert detector_conversion_spectrum.dtype == torch.float32, "The compute_conversion_spectrum method of the detector must return a torch.Tensor of type torch.float32"
        # should be 3D
        assert len(detector_conversion_spectrum.shape) == 3, "The compute_conversion_spectrum method of the detector must return a torch.Tensor of shape [nBatch x nDetectorChannel x nEnergy]"
        # the shape should be [nBatch x nDetectorChannel x nEnergy]
        nBatch = detector_conversion_spectrum.shape[0]

        # define an S2 tensor of shape [nBatch x nSourceChannel*nDetectorChannel x nSourceChannel*nDetectorChannel x nEnergy]
        S2 = torch.zeros([nBatch, nSourceChannel*nDetectorChannel, nSourceChannel*nDetectorChannel, nEnergy], dtype=torch.float32).to(device)
        for i in range(nSourceChannel):
            for j in range(nDetectorChannel):
                S2[:, i*nDetectorChannel+j, i*nDetectorChannel+j, :] = detector_conversion_spectrum[:, j, :]

        # reshape to [nBatch x nDetectorChannel*nSourceChannel x nSourceChannel*nDetectorChannel*nEnergy]
        S2 = S2.reshape(nBatch, nDetectorChannel*nSourceChannel, nSourceChannel*nDetectorChannel*nEnergy)
        return S2
    
    def compute_G(self):
        """
        This function computes the gain matrix, G.

        Returns:
            torch.Tensor: The gain matrix of shape [nBatch x nChannel x nOut_S2]. Type torch.float32. Shape [nBatch x nDetectorChannel*nSourceChannel x nDetectorChannel*nSourceChannel]. Description: The gain matrix.
        """

        # source.detector should be a SpectralDetector object
        # any SpectralDetector object must have a compute_gain method
        # and it must return a torch.Tensor of shape [nBatch x nDetectorChannel x nSourceChannel]
        # the final matrix G is of shape [nBatch x nChannel x nOut_S2] where nOut_S2 = nDetectorChannel*nSourceChannel.

        nSourceChannel = self.source.nSourceChannel
        nDetectorChannel = self.detector.nDetectorChannel

        # compute the detector gain
        detector_gain = self.detector.compute_gain()
        # torch tensor
        assert type(detector_gain) == torch.Tensor, "The compute_gain method of the detector must return a torch.Tensor"
        # should be float32
        assert detector_gain.dtype == torch.float32, "The compute_gain method of the detector must return a torch.Tensor of type torch.float32"
        # should be 3D
        assert len(detector_gain.shape) == 2, "The compute_gain method of the detector must return a torch.Tensor of shape [nBatch x nDetectorChannel x nSourceChannel]"
        # the shape should be [nBatch x nDetectorChannel]
        nBatch = detector_gain.shape[0]
        assert detector_gain.shape[1] == nDetectorChannel, "The compute_gain method of the detector must return a torch.Tensor of shape [nBatch x nDetectorChannel x nSourceChannel]"

        # define an G tensor of shape [nBatch x nDetectorChannel, nSourceChannel, nDetectorChannel, nSourceChannel]
        G = torch.zeros([nBatch, nDetectorChannel, nSourceChannel, nDetectorChannel, nSourceChannel], dtype=torch.float32).to(device)
        for i in range(nSourceChannel):
            for j in range(nDetectorChannel):
                G[:, j, i, j, i] = detector_gain[:, j]

        # reshape G to [nBatch x nChannel x nOut_S2]
        G = G.reshape([nBatch, nDetectorChannel*nSourceChannel, nDetectorChannel*nSourceChannel])

        return G

    def compute_S(self, returnMode=None):
        # ---------------------------
        # This class returns a batch of S mattrices as seen in the following forward model:
        #   y = S*exp(-Q*l)
        # where y is the detector measurement, 
        #       S is the spectral sensitivity matrix, 
        #       Q is the mass attenuation matrix, 
        #   and l are basis material density line integrals
        #
        # The S matrix can be decomposed as follows:
        #   S = G*S2*S1*S0
        # where S0 is the source spectral sensitivity
        #       S1 is the detector spectral sensitivity
        #       S2 is the deterministic spectral gain and/or integration
        #       G is the gain matrix
        #
        # we assume Q has the shape [nBatch x nEnergy x nMaterial]
        # so S should have the shape [nBatch x nChannel x nEnergy]
        #
        # ---------------------------
        # INPUTS:
        # ---------------------------
        #   returnMode
        # ---------------------------
        #       Name:
        #           Return Mode
        #       Type:
        #           str
        #       Default:
        #           None
        #       Options:
        #           None, 'S0', 'S1', 'S2', 'G', 'S1_S0', 'S2_S1_S0'
        #       Meaning:
        #           If None, then the full S matrix is returned.
        #           If 'S0', then only the S0 matrix is returned. 
        #           If 'S1', then only the S1 matrix is returned.
        #           If 'S2', then only the S2 matrix is returned.
        #           If 'G', then only the G matrix is returned.
        #           If 'S1_S0', the the matrix S1*S0 is returned.
        #           If 'S2_S1_S0', then the matrix S2*S1*S0 is returned.
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   S
        # ---------------------------
        #       Name:
        #           Total Spectral Sensitivity Spectrum
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nChannel x nEnergy]
        #       Output meaning:
        #           The total spectral sensitivity of the spectral imaging system from 0-1.
        #           Defined by the expected measurement during an air scan.
        #           A gain is applied such that the air scan measurements satisfy mean=variance for each channel. 
        # ---------------------------
        
        # compute S0
        S0 = self.compute_S0()
        # should be a tensor
        assert isinstance(S0, torch.Tensor), "Implementation Error, S0 should be a torch.Tensor"
        # should be float32
        assert S0.dtype == torch.float32, "Implementation Error, S0 should be torch.float32"
        # should have three dimensions
        assert S0.dim() == 3, "Implementation Error, S0 should have three dimensions"
        # should have shape [nBatch, nOut_S0, nEnergy]
        nBatch = S0.shape[0]
        nOut_S0 = S0.shape[1]
        nEnergy = S0.shape[2]

        if returnMode == "S0":
            return S0


        # compute S1
        S1 = self.compute_S1()
        # should be a tensor
        assert isinstance(S1, torch.Tensor), "Implementation Error, S1 should be a torch.Tensor"
        # should be float32
        assert S1.dtype == torch.float32, "Implementation Error, S1 should be torch.float32"
        # should have three dimensions
        assert S1.dim() == 3, "Implementation Error, S1 should have three dimensions"
        # first dimension should be nBatch
        assert S1.shape[0] == nBatch, "Implementation Error, S1 should have shape [nBatch, nOut_S1 , nOut_S0]"
        # third dimension should be nOut_S0
        assert S1.shape[2] == nOut_S0, "Implementation Error, S1 should have shape [nBatch, nOut_S1 , nOut_S0]"
        nOut_S1 = S1.shape[1]

        if returnMode == "S1":
            return S1

        S1_S0 = torch.matmul(S1, S0)

        if returnMode == "S1_S0":
            return S1_S0

        # compute S2
        S2 = self.compute_S2()
        # should be a tensor
        assert isinstance(S2, torch.Tensor), "Implementation Error, S2 should be a torch.Tensor"
        # should be float32
        assert S2.dtype == torch.float32, "Implementation Error, S2 should be torch.float32"
        # should have three dimensions
        assert S2.dim() == 3, "Implementation Error, S2 should have three dimensions"
        # first dimension should be nBatch
        assert S2.shape[0] == nBatch, "Implementation Error, S1 should have shape [nBatch, nOut_S2 , nOut_S1]"
        # third dimension should be nOut_S1
        assert S2.shape[2] == nOut_S1, "Implementation Error, S1 should have shape [nBatch, nOut_S2 , nOut_S1]"
        nOut_S2 = S2.shape[1]

        S2_S1_S0 = torch.matmul(S2, S1_S0)

        if returnMode == "S2_S1_S0":
            return S2_S1_S0
        
        if returnMode == "S2":
            return S2

        # compute G
        G = self.compute_G()
        # should be a tensor
        assert isinstance(G, torch.Tensor), "Implementation Error, G should be a torch.Tensor"
        # should be float32
        assert G.dtype == torch.float32, "Implementation Error, G should be torch.float32"
        # should have three dimensions
        assert G.dim() == 3, "Implementation Error, G should have three dimensions"
        # first dimension should be nBatch
        assert G.shape[0] == nBatch, "Implementation Error, G should have shape [nBatch, nChannel , nOut_S2]"
        # third dimension should be nOut_S2
        assert G.shape[2] == nOut_S2, "Implementation Error, G should have shape [nBatch, nChannel , nOut_S2]"

        if returnMode == "G":
            return G       

        S = torch.matmul(G, S2_S1_S0)

        return S

    def forward(self, attenuator_transmission_probability_spectrum):
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   attenuator_transmission_probability_spectrum
        # ---------------------------
        #       Name:
        #           attenuator Transmission Probability Spectrum
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nEnergy x 1]
        #       Output Meaning:
        #           The probability that an x-ray photon will pass through the
        #           attenuator and reach the detector.
        #           this should be the output of exp(-Q*l)
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   measurements
        # ---------------------------
        #       Name:
        #           Measurements
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nChannel x 1]
        #       Output Meaning:
        #           The expected measurement of the spectral imaging system.
        # ---------------------------

        # check that the input is a tensor
        assert isinstance(attenuator_transmission_probability_spectrum, torch.Tensor), "The input attenuator_transmission_probability_spectrum should be a torch.Tensor"
        # check that the input is float32
        assert attenuator_transmission_probability_spectrum.dtype == torch.float32, "The input attenuator_transmission_probability_spectrum should be torch.float32"
        # check that the input has three dimensions
        assert attenuator_transmission_probability_spectrum.dim() == 3, "The input attenuator_transmission_probability_spectrum should have three dimensions"
        # check that the input has the correct shape
        assert attenuator_transmission_probability_spectrum.shape[2] == 1, "The input attenuator_transmission_probability_spectrum should have shape [nBatch x nEnergy x 1]"
        nBatch = attenuator_transmission_probability_spectrum.shape[0]
        nEnergy = attenuator_transmission_probability_spectrum.shape[1]

        # compute the total sensitivity spectrum of the spectral imaging system
        S = self.compute_S()

        # check for compatibility between the input and the sensitivity spectrum
        assert S.shape[0] == nBatch, "The input attenuator_transmission_probability_spectrum and spectral sensitivity matrix, S, should have the same number of batches but S has shape {} and the input has shape {}".format(S.shape, attenuator_transmission_probability_spectrum.shape)
        assert S.shape[1] == nEnergy, "The input attenuator_transmission_probability_spectrum and spectral sensitivity matrix, S, should have the same number of energy bins but S has shape {} and the input has shape {}".format(S.shape, attenuator_transmission_probability_spectrum.shape)
        
        # compute the measurements
        measurements = torch.matmul(S, attenuator_transmission_probability_spectrum)

        return measurements


# make a BasisMaterials class
# the inputs should be the a list of atomic numbers of size [nBatch, nEnergy, nMaterial]
# this class doesn't really do anything it just returns the input as Q

class BasisMaterials(nn.Module):
    def __init__(self, basis_mass_attenuation_spectra):
        super(BasisMaterials, self).__init__()
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   basis_mass_attenuation_spectra
        # ---------------------------
        #       Name:
        #           Basis Mass Attenuation Spectra
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nEnergy x nMaterial]
        #       Output Meaning:
        #           The mass attenuation coefficient of the basis materials (in units of g/mm^2).
        # ---------------------------

        # check that the input is a tensor
        assert isinstance(basis_mass_attenuation_spectra, torch.Tensor), "The input basis_mass_attenuation_spectra should be a torch.Tensor"
        # check that the input is float32
        assert basis_mass_attenuation_spectra.dtype == torch.float32, "The input basis_mass_attenuation_spectra should be torch.float32"
        # check that the input has three dimensions
        assert basis_mass_attenuation_spectra.dim() == 3, "The input basis_mass_attenuation_spectra should have three dimensions"
        # assume that the first dimension is nBatch
        self.nBatch = basis_mass_attenuation_spectra.shape[0]
        # assume that the second dimension is nEnergy
        self.nEnergy = basis_mass_attenuation_spectra.shape[1]
        # assume that the third dimension is nMaterial
        self.nMaterial = basis_mass_attenuation_spectra.shape[2]

        self.basis_mass_attenuation_spectra = basis_mass_attenuation_spectra

    def compute_Q(self):
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   Q
        # ---------------------------
        #       Name:
        #           Q
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nEnergy x nMaterial]
        #       Output Meaning:
        #           The mass attenuation spectrum of the basis materials (in units of mm^2/g).

        return self.basis_mass_attenuation_spectra

    def forward(self, basis_density_line_integrals):
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   basis_density_line_integrals
        # ---------------------------
        #       Name:
        #           Basis Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Output Meaning:
        #           The line integral of the density of the basis materials (in units of g/mm^2).
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   unitless_attenuation
        # ---------------------------
        #       Name:
        #           Unitless Attenuation
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nEnergy x nPixel]
        #       Output Meaning:
        #           The unitless attenuation of the basis materials (in units of mm^2/g).
        # ---------------------------


        # check that the input is a tensor
        assert isinstance(basis_density_line_integrals, torch.Tensor), "The input basis_density_line_integrals should be a torch.Tensor"
        # check that the input is float32
        assert basis_density_line_integrals.dtype == torch.float32, "The input basis_density_line_integrals should be torch.float32"
        # check that the input has three dimensions
        assert basis_density_line_integrals.dim() == 3, "The input basis_density_line_integrals should have three dimensions"
        # check that the input has the correct shape
        assert basis_density_line_integrals.shape[0] == self.nBatch, "The input basis_density_line_integrals should have shape [nBatch x nMaterial x 1] but the input has shape {}".format(basis_density_line_integrals.shape)
        assert basis_density_line_integrals.shape[1] == self.nMaterial, "The input basis_density_line_integrals should have shape [nBatch x nMaterial x 1] but the input has shape {}".format(basis_density_line_integrals.shape)
        # assume that the third dimension is nPixel
        self.nPixel = basis_density_line_integrals.shape[2]


        # compute the mass attenuation spectra
        Q = self.compute_Q()

        # compute the matrix product
        unitless_attenuation = torch.bmm(Q, basis_density_line_integrals)
        assert unitless_attenuation.shape == (self.nBatch, self.nEnergy, self.nPixel), "Something went wrong. The output unitless_attenuation should have shape [nBatch x nEnergy x nPixel] but the output has shape {}".format(unitless_attenuation.shape)

        return unitless_attenuation

# make a ElementalBasisMaterials class
# the inputs should be the a list of atomic numbers of size [nBatch, nElements]
#            and the list of relative density of each element in each material of size [nBatch, nElements, nMaterial]
# it should have a method called compute Q that returns the mass attenuation coefficient for the basis materials

class ElementalBasisMaterials(BasisMaterials):
    def __init__(self, atomic_number, relative_density):

        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   atomic_number
        # ---------------------------
        #       Name:
        #           Atomic Numbers
        #       Type:
        #           torch.Tensor (torch.long)
        #       Shape:
        #           [nBatch x nElements]
        #       Output Meaning:
        #           The atomic numbers of the elements in the basis materials.
        # ---------------------------
        # ---------------------------
        #   relative_density
        # ---------------------------
        #       Name:
        #           Relative Density
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nElements]
        #       Output Meaning:
        #           The relative density of each element in each material.
        # ---------------------------

        # check that the input is a tensor
        assert isinstance(atomic_number, torch.Tensor), "The input atomic_number should be a torch.Tensor"
        # check that the input is float32
        assert atomic_number.dtype == torch.long, "The input atomic_number should be torch.long (long integer)"
        # check that the input has three dimensions
        assert atomic_number.dim() == 2, "The input atomic_number should have two dimensions"
        # assume shape [nBatch, nElements]
        self.nBatch = atomic_number.shape[0]
        self.nElements = atomic_number.shape[1]

        # check that the input is a tensor
        assert isinstance(relative_density, torch.Tensor), "The input relative_density should be a torch.Tensor"
        # check that the input is float32
        assert relative_density.dtype == torch.float32, "The input relative_density should be torch.float32"
        # check that the input has three dimensions
        assert relative_density.dim() == 3, "The input relative_density should have three dimensions"
        # check that the input has the correct shape
        assert relative_density.shape[0] == self.nBatch, "The input relative_density should have shape [nBatch x nMaterial x nElements ]"
        assert relative_density.shape[2] == self.nElements, "The input relative_density should have shape [nBatch x nMaterial x nElements]"
        self.nMaterial = relative_density.shape[1]
        self.nEnergy = photon_kev.shape[0]

        # normalize relative density so that the sum of the relative density of each element in each material is 1
        # relative_density = relative_density/torch.sum(relative_density, dim=1, keepdim=True)

        # massAttenuationSpectra = torch.zeros([150,91])

        basis_mass_attenuation_spectra = torch.zeros([self.nBatch, self.nEnergy, self.nMaterial], dtype=torch.float32).to(device)
        for iMaterial in range(self.nMaterial):
            for iElement in range(self.nElements):
                basis_mass_attenuation_spectra[:,:,iMaterial] += massAttenuationSpectra[:,atomic_number[:,iElement]].permute(1,0)*relative_density[:,iMaterial,iElement].unsqueeze(1)

        # initialize the parent class BasisMaterials
        super().__init__(basis_mass_attenuation_spectra)


class SpectralImagingPhysicsModel(nn.Module):
    def __init__(
            self,
            spectral_imaging_system,
            basis_materials,
            ):
            # ---------------------------
            # INPUTS:
            # ---------------------------
            # ---------------------------
            #   spectral_imaging_system
            # ---------------------------
            #       Name:
            #           Spectral Imaging System
            #       Type:
            #           SpectralImagingSystem
            #       Output Meaning:
            #           The spectral imaging system.
            #           used to compute S
            #           S should have units of 
            #               measurement counts per photon 
            #               transmitted through the attenuator
            # ---------------------------
            # ---------------------------
            #   basis_materials
            # ---------------------------
            #       Name:
            #           Basis Materials
            #       Type:
            #           BasisMaterials
            #       Output Meaning:
            #           The basis materials.
            #           used to compute Q
            #           Q should be mass attenuation spectra of the basis materials
            #           Q should have units of mm^2/g
            # ---------------------------

            super(SpectralImagingPhysicsModel, self).__init__()

            assert isinstance(spectral_imaging_system, SpectralImagingSystem), "The input spectral_imaging_system should be a SpectralImagingSystem object"
            assert isinstance(basis_materials, BasisMaterials), "The input basis_materials should be a BasisMaterials object"

            self.spectral_imaging_system = spectral_imaging_system
            self.basis_materials = basis_materials

            self.nMaterial = self.basis_materials.nMaterial
            self.nChannel = self.spectral_imaging_system.nChannel

    def compute_S(self,**kwargs):
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   S
        # ---------------------------
        #       Name:
        #           Spectral Sensitivity Matrix
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nChannel x nEnergy]
        #       Meaning:
        #           The spectral sensitivity matrix.
        #           S should have units of
        #               measurement counts per photon
        #               transmitted through the attenuator
        # ---------------------------

        # compute the spectral sensitivity matrix
        S = self.spectral_imaging_system.compute_S(**kwargs)

        return S

    def compute_Q(self,**kwargs):      
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   Q
        # ---------------------------
        #       Name:
        #           Basis Material Mass Attenuation Spectra
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nEnergy x nMaterial]
        #       Meaning:
        #           The basis material mass attenuation spectra.
        #           Q should be mass attenuation
        #           spectra of the basis materials
        #           Q should have units of mm^2/g
        # ---------------------------
        
        # compute the basis material mass attenuation spectra
        Q = self.basis_materials.compute_Q(**kwargs)

        return Q
    def compute_Sigma_y(self, l):
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   l
        # ---------------------------
        #       Name:
        #           Material Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The material density line integrals.
        #           l should have units of g/mm^2
        # ---------------------------
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   Sigma_y
        # ---------------------------
        #       Name:
        #           Measurement Noise Covariance Matrix
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nChannel x nChannel x nPixel]
        #       Meaning:
        #           The measurement noise covariance matrix.
        #           Sigma_y should have units of
        #               measurement counts per photon
        #               transmitted through the attenuator
        # ---------------------------

        # check the input
        assert isinstance(l, torch.Tensor), "The input l should be a torch.Tensor"
        assert l.dtype == torch.float32, "The input l should be a torch.Tensor of type torch.float32"
        assert len(l.shape) == 3, "The input l should have shape [nBatch x nMaterial x nPixel]"
        
        nBatch = l.shape[0]
        nMaterial = l.shape[1]
        nPixel = l.shape[2]

        # compute Q
        Q = self.compute_Q()
        nEnergy = Q.shape[1]
        nMaterial = Q.shape[2]
        assert Q.shape == torch.Size([nBatch,nEnergy,nMaterial]), "The shape of Q should be [nBatch x nEnergy x nMaterial]"

        # compute the S1_S0 matrix
        S1_S0 = self.compute_S(returnMode='S1_S0')
        # at this point, S1_S0 has shape [nBatch x nOut_S1 x nEnergy]
        nOut_S1 = S1_S0.shape[1]
        assert S1_S0.shape == torch.Size([nBatch,nOut_S1,nEnergy]), "The shape of S1_S0 should be [nBatch x nOut_S1 x nEnergy]"
        
        # compute the S2 matrix
        S2 = self.compute_S(returnMode='S2')
        # at this point, S2 has shape [nBatch x nOut_S2 x nOut_S1]
        nOut_S2 = S2.shape[1]
        assert S2.shape == torch.Size([nBatch,nOut_S2,nOut_S1]), "The shape of S2 should be [nBatch x nOut_S2 x nOut_S1]"
        
        # compute the G matrix
        G = self.compute_S(returnMode='G')
        # at this point, G has shape [nBatch x nChannel x nOut_S2]
        nChannel = G.shape[1]
        assert G.shape == torch.Size([nBatch,nChannel,nOut_S2]), "The shape of G should be [nBatch x nChannel x nOut_S2]"
        
        # compute z = S1_S0 * exp(-Q*l)
        # l has shape [nBatch x nMaterial x nPixel]
        z = torch.bmm(Q,l)
        # at this point, z has shape [nBatch x nEnergy x nPixel]
        z = torch.exp(-z)
        # at this point, z has shape [nBatch x nEnergy x nPixel]
        z = torch.bmm(S1_S0,z)
        # at this point, z has shape [nBatch x nOut_S1 x nPixel]
        # permute z to have shape [nBatch x nPixel x nOut_S1]
        z = z.permute(0,2,1)
        # at this point, z has shape [nBatch x nPixel x nOut_S1]
        assert z.shape == torch.Size([nBatch,nPixel,nOut_S1]), "The shape of z should be [nBatch x nPixel x nOut_S1]"
        # combine the batch and pixel dimensions
        z = z.reshape(nBatch*nPixel,nOut_S1)

        # z is the number of photons interacting with the detector
        # so it should be poisson distributed 
        # so covariance is diagonal with variance equal to mean = z
        # so use diag embed to make the diagonal matrix
        Sigma_z = torch.diag_embed(z)
        # at this point, Sigma_z has shape [nBatch*nPixel x nOut_S1 x nOut_S1]
        assert Sigma_z.shape == torch.Size([nBatch*nPixel,nOut_S1,nOut_S1]), "The shape of Sigma_z should be [nBatch*nPixel x nOut_S1 x nOut_S1]"
        
        # assume S2 is the same for all pixels for now
        # expand S2 to have shape [nBatch x nPixel x nOut_S2 x nOut_S1]
        S2 = S2.unsqueeze(1).expand(nBatch,nPixel,nOut_S2,nOut_S1)
        # at this point, S2 has shape [nBatch x nPixel x nOut_S2 x nOut_S1]
        assert S2.shape == torch.Size([nBatch,nPixel,nOut_S2,nOut_S1]), "The shape of S2 should be [nBatch x nPixel x nOut_S2 x nOut_S1]"
        # combine nBatch and nPixel dimensions
        S2 = S2.reshape(nBatch*nPixel,nOut_S2,nOut_S1)
        # at this point, S2 has shape [nBatch*nPixel x nOut_S2 x nOut_S1]
        assert S2.shape == torch.Size([nBatch*nPixel,nOut_S2,nOut_S1]), "The shape of S2 should be [nBatch*nPixel x nOut_S2 x nOut_S1]"

        # assume G is the same for all pixels for now
        # expand G to have shape [nBatch x nPixel x nChannel x nOut_S2]
        G = G.unsqueeze(1).expand(nBatch,nPixel,nChannel,nOut_S2)
        # at this point, G has shape [nBatch x nPixel x nChannel x nOut_S2]
        assert G.shape == torch.Size([nBatch,nPixel,nChannel,nOut_S2]), "The shape of G should be [nBatch x nPixel x nChannel x nOut_S2]"
        # combine nBatch and nPixel dimensions
        G = G.reshape(nBatch*nPixel,nChannel,nOut_S2)
        # at this point, G has shape [nBatch*nPixel x nChannel x nOut_S2]
        assert G.shape == torch.Size([nBatch*nPixel,nChannel,nOut_S2]), "The shape of G should be [nBatch*nPixel x nChannel x nOut_S2]"


        # y = G * S2 * z 
        # where G and S2 are deterministic gain
        # so Sigma_y = G * S2 * Sigma_z * S2^T * G^T
        Sigma_y = G.transpose(1,2)
        Sigma_y = torch.bmm(S2.transpose(1,2),Sigma_y)
        Sigma_y = torch.bmm(Sigma_z,Sigma_y)
        Sigma_y = torch.bmm(S2,Sigma_y)
        Sigma_y = torch.bmm(G,Sigma_y)
        # at this point, Sigma_y has shape [nBatch*nPixel x nChannel x nChannel]
        assert Sigma_y.shape == torch.Size([nBatch*nPixel,nChannel,nChannel]), "The shape of Sigma_y should be [nBatch*nPixel x nChannel x nChannel]"

        # reshape to have shape [nBatch x nPixel x nChannel x nChannel]
        Sigma_y = Sigma_y.reshape(nBatch,nPixel,nChannel,nChannel)
        # at this point, Sigma_y has shape [nBatch x nPixel x nChannel x nChannel]
        assert Sigma_y.shape == torch.Size([nBatch,nPixel,nChannel,nChannel]), "The shape of Sigma_y should be [nBatch x nPixel x nChannel x nChannel]"
        # permute to have shape [nBatch x nChannel x nChannel x nPixel]
        Sigma_y = Sigma_y.permute(0,2,3,1)
        # at this point, Sigma_y has shape [nBatch x nChannel x nChannel x nPixel]
        assert Sigma_y.shape == torch.Size([nBatch,nChannel,nChannel,nPixel]), "The shape of Sigma_y should be [nBatch x nChannel x nChannel x nPixel]"

        return Sigma_y

    def compute_grad(self,y, l_hat):
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   y
        # ---------------------------
        #       Name:
        #           Measurements
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nChannel x nPixel]
        #       Meaning:
        #           The measured detector counts
        #           (not photon counts, includes gain)
        # ---------------------------
        #   l_hat
        # ---------------------------
        #       Name:
        #           Estimated Basis Material Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The line integral of the estimated basis material density (g/mm2)
        # ---------------------------
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   grad
        # ---------------------------
        #       Name:
        #           Fisher Information Matrix
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The Fisher Information Matrix
        #           The inverse of the Cramer-Rao Lower Bound (CRLB) covariance matrix
        #           Relevant for computing the detectability of Maximum Likelihood Estimators (MLE)
        #           It is not the same as Shannon entropy which is commonly called "information" 
        #           The inverse of a covariance matrix is a precision matrix
        #           It describes how precise the measurements are for detecting any combination of materials
        # ---------------------------

        # check the input
        assert isinstance(y, torch.Tensor), 'y must be a torch.Tensor'
        assert y.dtype == torch.float32, 'y must be a torch.float32'
        assert y.ndim == 3, 'y must be a 3D tensor'
        nBatch = y.shape[0]
        nChannel = y.shape[1]
        nPixel = y.shape[2]
        # permute the y tensor to be [nBatch x nPixel x nChannel]
        y = y.permute(0,2,1)
        # reshape y to be [nBatch*nPixel x nChannel x 1]
        y = y.reshape(nBatch*nPixel,nChannel,1)

        # check the other input
        assert isinstance(l_hat, torch.Tensor), 'l_hat must be a torch.Tensor'
        assert l_hat.dtype == torch.float32, 'l_hat must be a torch.float32'
        assert l_hat.ndim == 3, 'l_hat must be a 3D tensor'
        assert l_hat.shape[0] == nBatch, 'l_hat must have the same number of batches as y'
        assert l_hat.shape[2] == nPixel, 'l_hat must have the same number of pixels as y'
        nMaterial = l_hat.shape[1]

        # compute Q
        Q = self.compute_Q()
        nEnergy = Q.shape[1]
        nMaterial = Q.shape[2]
        assert Q.shape == torch.Size([nBatch,nEnergy,nMaterial]), "The shape of Q should be [nBatch x nEnergy x nMaterial]"

        # compute the D matrix by applying Q to l_hat
        D = torch.exp(-torch.bmm(Q,l_hat))
        # at this point, D has shape [nBatch x nEnergy x nPixel]
        assert D.shape == torch.Size([nBatch,nEnergy,nPixel]), "The shape of D should be [nBatch x nEnergy x nPixel]"
        # permute D to have shape [nBatch x nPixel x nEnergy]
        D = D.permute(0,2,1)
        # at this point, D has shape [nBatch x nPixel x nEnergy]
        assert D.shape == torch.Size([nBatch,nPixel,nEnergy]), "The shape of D should be [nBatch x nPixel x nEnergy]"
        # use diag embedding to make D have shape [nBatch x nPixel x nEnergy x nEnergy]
        D = torch.diag_embed(D)
        # at this point, D has shape [nBatch x nPixel x nEnergy x nEnergy]
        assert D.shape == torch.Size([nBatch,nPixel,nEnergy,nEnergy]), "The shape of D should be [nBatch x nPixel x nEnergy x nEnergy]"
        # combine nBatch and nPixel dimensions
        D = D.reshape(nBatch*nPixel,nEnergy,nEnergy)
        # at this point, D has shape [nBatch*nPixel x nEnergy x nEnergy]
        assert D.shape == torch.Size([nBatch*nPixel,nEnergy,nEnergy]), "The shape of D should be [nBatch*nPixel x nEnergy x nEnergy]"

        
        # compute the S matrix
        S = self.compute_S()
        # at this point, S has shape [nBatch x nChannel x nEnergy]
        nChannel = S.shape[1]
        assert S.shape == torch.Size([nBatch,nChannel,nEnergy]), "The shape of S should be [nBatch x nChannel x nEnergy]"


        # assume Q is the same for all pixels for now
        # expand Q to have shape [nBatch x nPixel x nEnergy x nMaterial]
        Q = Q.unsqueeze(1).expand(nBatch,nPixel,nEnergy,nMaterial)
        # at this point, Q has shape [nBatch x nPixel x nEnergy x nMaterial]
        assert Q.shape == torch.Size([nBatch,nPixel,nEnergy,nMaterial]), "The shape of Q should be [nBatch x nPixel x nEnergy x nMaterial]"
        # combine nBatch and nPixel dimensions
        Q = Q.reshape(nBatch*nPixel,nEnergy,nMaterial)
        # at this point, Q has shape [nBatch*nPixel x nEnergy x nMaterial]
        assert Q.shape == torch.Size([nBatch*nPixel,nEnergy,nMaterial]), "The shape of Q should be [nBatch*nPixel x nEnergy x nMaterial]"


        # assume S is the same for all pixels for now
        # expand S to have shape [nBatch x nPixel x nChannel x nEnergy]
        S = S.unsqueeze(1).expand(nBatch,nPixel,nChannel,nEnergy)
        # at this point, S has shape [nBatch x nPixel x nChannel x nEnergy]
        assert S.shape == torch.Size([nBatch,nPixel,nChannel,nEnergy]), "The shape of S should be [nBatch x nPixel x nChannel x nEnergy]"
        # combine nBatch and nPixel dimensions
        S = S.reshape(nBatch*nPixel,nChannel,nEnergy)
        # at this point, S has shape [nBatch*nPixel x nChannel x nEnergy]
        assert S.shape == torch.Size([nBatch*nPixel,nChannel,nEnergy]), "The shape of S should be [nBatch*nPixel x nChannel x nEnergy]"

        # compute the measurement covariance
        Sigma_y = self.compute_Sigma_y(l_hat)
        # Sigma_y should have shape [nBatch x nChannel x nChannel x nPixel]
        assert Sigma_y.shape == torch.Size([nBatch,nChannel,nChannel,nPixel]), "The shape of Sigma_y should be [nBatch x nChannel x nChannel x nPixel]"
        # permute Sigma_y to have shape [nBatch x nPixel x nChannel x nChannel]
        Sigma_y = Sigma_y.permute(0,3,1,2)
        # at this point, Sigma_y has shape [nBatch x nPixel x nChannel x nChannel]
        assert Sigma_y.shape == torch.Size([nBatch,nPixel,nChannel,nChannel]), "The shape of Sigma_y should be [nBatch x nPixel x nChannel x nChannel]"
        # combine nBatch and nPixel dimensions
        Sigma_y = Sigma_y.reshape(nBatch*nPixel,nChannel,nChannel)
        # at this point, Sigma_y has shape [nBatch*nPixel x nChannel x nChannel]
        assert Sigma_y.shape == torch.Size([nBatch*nPixel,nChannel,nChannel]), "The shape of Sigma_y should be [nBatch*nPixel x nChannel x nChannel]"
        # compute the inverse of Sigma_y
        invSigma_y = torch.linalg.inv(Sigma_y)
        # at this point, invSigma_y has shape [nBatch*nPixel x nChannel x nChannel]
        assert invSigma_y.shape == torch.Size([nBatch*nPixel,nChannel,nChannel]), "The shape of invSigma_y should be [nBatch*nPixel x nChannel x nChannel]"


        # compute y_bar
        y_bar = self.forward(l_hat)
        # y_bar should have shape [nBatch x nChannel x nPixel]
        assert y_bar.shape == torch.Size([nBatch,nChannel,nPixel]), "The shape of y_bar should be [nBatch x nChannel x nPixel]"
        # permute y_bar to have shape [nBatch x nPixel x nChannel]
        y_bar = y_bar.permute(0,2,1)
        # make it [nBatch*nPixel x nChannel x 1]
        y_bar = y_bar.reshape(nBatch*nPixel,nChannel,1)
        assert y_bar.shape == torch.Size([nBatch*nPixel,nChannel,1]), "The shape of y_bar should be [nBatch*nPixel x nChannel x 1]"


        # Now we need to compute the gradient
        # the formula for the gradient is:
        #         grad = torch.matmul(torch.matmul(S_D_Q.T,inv_Simgay),(y - y_bar))
        #   grad = Q^T * D^T * S^T * invSigma_y * (y - y_bar)
        # we will do it step by step with batched matrix multiply

        grad = y - y_bar
        grad = torch.bmm(invSigma_y, grad)
        grad = torch.bmm(S.transpose(1,2), grad)
        grad = torch.bmm(D.transpose(1,2), grad)
        grad = torch.bmm(Q.transpose(1,2), grad)
        
        # reshape grad to have shape [nBatch x nPixel x nMaterial]
        grad = grad.reshape(nBatch,nPixel,nMaterial)
        # at this point, grad has shape [nBatch x nPixel x nMaterial]
        assert grad.shape == torch.Size([nBatch,nPixel,nMaterial]), "The shape of grad should be [nBatch x nPixel x nMaterial]"
        # permute grad to have shape [nBatch x nMaterial x nPixel]
        grad = grad.permute(0,2,1)
        # at this point, grad has shape [nBatch x nMaterial x nPixel]
        assert grad.shape == torch.Size([nBatch,nMaterial,nPixel]), "The shape of grad should be [nBatch x nMaterial x nPixel]"

        return grad
    
    def compute_F(self, l_background):
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   l_background
        # ---------------------------
        #       Name:
        #           Background Basis Material Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The background basis material density line integrals.
        #           l_background should have units of g/mm^2
        # ---------------------------
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   F
        # ---------------------------
        #       Name:
        #           Fisher Information Matrix
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nMaterial x nPixel]
        #       Meaning:
        #           The Fisher Information Matrix
        #           The inverse of the Cramer-Rao Lower Bound (CRLB) covariance matrix
        #           Relevant for computing the detectability of Maximum Likelihood Estimators (MLE)
        #           It is not the same as Shannon entropy which is commonly called "information" 
        #           The inverse of a covariance matrix is a precision matrix
        #           It describes how precise the measurements are for detecting any combination of materials
        # ---------------------------
    

        # check the input
        assert isinstance(l_background, torch.Tensor), "The input l_background should be a torch.Tensor"
        assert l_background.dtype == torch.float32, "The input l_background should be a torch.float32"
        assert len(l_background.shape) == 3, "The input l_background should have 3 dimensions"
        nBatch = l_background.shape[0]
        nMaterial = l_background.shape[1]
        nPixel = l_background.shape[2]

        # compute Q
        Q = self.compute_Q()
        nEnergy = Q.shape[1]
        nMaterial = Q.shape[2]
        assert Q.shape == torch.Size([nBatch,nEnergy,nMaterial]), "The shape of Q should be [nBatch x nEnergy x nMaterial]"

        # compute the D matrix by applying Q to l_background
        D = torch.exp(-torch.bmm(Q,l_background))
        # at this point, D has shape [nBatch x nEnergy x nPixel]
        assert D.shape == torch.Size([nBatch,nEnergy,nPixel]), "The shape of D should be [nBatch x nEnergy x nPixel]"
        # permute D to have shape [nBatch x nPixel x nEnergy]
        D = D.permute(0,2,1)
        # at this point, D has shape [nBatch x nPixel x nEnergy]
        assert D.shape == torch.Size([nBatch,nPixel,nEnergy]), "The shape of D should be [nBatch x nPixel x nEnergy]"
        # use diag embedding to make D have shape [nBatch x nPixel x nEnergy x nEnergy]
        D = torch.diag_embed(D)
        # at this point, D has shape [nBatch x nPixel x nEnergy x nEnergy]
        assert D.shape == torch.Size([nBatch,nPixel,nEnergy,nEnergy]), "The shape of D should be [nBatch x nPixel x nEnergy x nEnergy]"
        # combine nBatch and nPixel dimensions
        D = D.reshape(nBatch*nPixel,nEnergy,nEnergy)
        # at this point, D has shape [nBatch*nPixel x nEnergy x nEnergy]
        assert D.shape == torch.Size([nBatch*nPixel,nEnergy,nEnergy]), "The shape of D should be [nBatch*nPixel x nEnergy x nEnergy]"

        
        # compute the S matrix
        S = self.compute_S()
        # at this point, S has shape [nBatch x nChannel x nEnergy]
        nChannel = S.shape[1]
        assert S.shape == torch.Size([nBatch,nChannel,nEnergy]), "The shape of S should be [nBatch x nChannel x nEnergy]"


        # assume Q is the same for all pixels for now
        # expand Q to have shape [nBatch x nPixel x nEnergy x nMaterial]
        Q = Q.unsqueeze(1).expand(nBatch,nPixel,nEnergy,nMaterial)
        # at this point, Q has shape [nBatch x nPixel x nEnergy x nMaterial]
        assert Q.shape == torch.Size([nBatch,nPixel,nEnergy,nMaterial]), "The shape of Q should be [nBatch x nPixel x nEnergy x nMaterial]"
        # combine nBatch and nPixel dimensions
        Q = Q.reshape(nBatch*nPixel,nEnergy,nMaterial)
        # at this point, Q has shape [nBatch*nPixel x nEnergy x nMaterial]
        assert Q.shape == torch.Size([nBatch*nPixel,nEnergy,nMaterial]), "The shape of Q should be [nBatch*nPixel x nEnergy x nMaterial]"


        # assume S is the same for all pixels for now
        # expand S to have shape [nBatch x nPixel x nChannel x nEnergy]
        S = S.unsqueeze(1).expand(nBatch,nPixel,nChannel,nEnergy)
        # at this point, S has shape [nBatch x nPixel x nChannel x nEnergy]
        assert S.shape == torch.Size([nBatch,nPixel,nChannel,nEnergy]), "The shape of S should be [nBatch x nPixel x nChannel x nEnergy]"
        # combine nBatch and nPixel dimensions
        S = S.reshape(nBatch*nPixel,nChannel,nEnergy)
        # at this point, S has shape [nBatch*nPixel x nChannel x nEnergy]
        assert S.shape == torch.Size([nBatch*nPixel,nChannel,nEnergy]), "The shape of S should be [nBatch*nPixel x nChannel x nEnergy]"

        # compute the measurement covariance
        Sigma_y = self.compute_Sigma_y(l_background)
        # Sigma_y should have shape [nBatch x nChannel x nChannel x nPixel]
        assert Sigma_y.shape == torch.Size([nBatch,nChannel,nChannel,nPixel]), "The shape of Sigma_y should be [nBatch x nChannel x nChannel x nPixel]"
        # permute Sigma_y to have shape [nBatch x nPixel x nChannel x nChannel]
        Sigma_y = Sigma_y.permute(0,3,1,2)
        # at this point, Sigma_y has shape [nBatch x nPixel x nChannel x nChannel]
        assert Sigma_y.shape == torch.Size([nBatch,nPixel,nChannel,nChannel]), "The shape of Sigma_y should be [nBatch x nPixel x nChannel x nChannel]"
        # combine nBatch and nPixel dimensions
        Sigma_y = Sigma_y.reshape(nBatch*nPixel,nChannel,nChannel)
        # at this point, Sigma_y has shape [nBatch*nPixel x nChannel x nChannel]
        assert Sigma_y.shape == torch.Size([nBatch*nPixel,nChannel,nChannel]), "The shape of Sigma_y should be [nBatch*nPixel x nChannel x nChannel]"
        # compute the inverse of Sigma_y
        invSigma_y = torch.linalg.inv(Sigma_y)
        # at this point, invSigma_y has shape [nBatch*nPixel x nChannel x nChannel]
        assert invSigma_y.shape == torch.Size([nBatch*nPixel,nChannel,nChannel]), "The shape of invSigma_y should be [nBatch*nPixel x nChannel x nChannel]"

        # Now we need to compute the Fisher information matrix
        # the formula for the Fisher information matrix is:
        #   F = Q^T * D^T * S^T * inv(Sigma_y) * S * D * Q
        # we will do it step by step with batched matrix multiply

        F = Q.clone()
        F = torch.bmm(D, F)
        F = torch.bmm(S,F)
        F = torch.bmm(invSigma_y, F)
        F = torch.bmm(S.transpose(1,2), F)
        F = torch.bmm(D.transpose(1,2), F)
        F = torch.bmm(Q.transpose(1,2), F)

        # at this point, F has shape [nBatch*nPixel x nMaterial x nMaterial]
        assert F.shape == torch.Size([nBatch*nPixel,nMaterial,nMaterial]), "The shape of F should be [nBatch*nPixel x nMaterial x nMaterial]"
        # reshape F to have shape [nBatch x nPixel x nMaterial x nMaterial]
        F = F.reshape(nBatch,nPixel,nMaterial,nMaterial)
        # at this point, F has shape [nBatch x nPixel x nMaterial x nMaterial]
        assert F.shape == torch.Size([nBatch,nPixel,nMaterial,nMaterial]), "The shape of F should be [nBatch x nPixel x nMaterial x nMaterial]"
        # permute F to have shape [nBatch x nMaterial x nMaterial x nPixel]
        F = F.permute(0,3,2,1)
        # at this point, F has shape [nBatch x nMaterial x nMaterial x nPixel]
        assert F.shape == torch.Size([nBatch,nMaterial,nMaterial,nPixel]), "The shape of F should be [nBatch x nMaterial x nMaterial x nPixel]"

        return F

    def forward(self, l):
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   l
        # ---------------------------
        #       Name:
        #           Basis Material Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The line integral of the basis material density (g/mm2)
        # ---------------------------
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   y_bar
        # ---------------------------
        #       Name:
        #           Expected Measurements
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nChannel x nPixel]
        #       Meaning:
        #           The measured detector counts 
        #           (not photon counts, includes gain)
        # ---------------------------
        
        # compute the spectral design matrix
        S = self.compute_S()

        # compute the basis material mass attenuation
        Q = self.compute_Q()

        # compute the y vector representing the measured photon counts

        z = l.clone()
        z = torch.bmm(Q,l)
        z = torch.exp(-z)
        y_bar = torch.bmm(S,z)

        return y_bar

    def forward_with_noise(self, l):
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   l
        # ---------------------------
        #       Name:
        #           Basis Material Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The line integral of the basis material density (g/mm2)
        # ---------------------------
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   y
        # ---------------------------
        #       Name:
        #           Measurements with Noise
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nChannel]
        #       Meaning:
        #           The measured detector counts 
        #           (not photon counts, includes gain)
        # ---------------------------
        
        # compute the spectral design matrix
        S1_S0 = self.compute_S(returnMode='S1_S0')
        S2 = self.compute_S(returnMode='S2')
        G = self.compute_S(returnMode='G')

        # compute the basis material mass attenuation
        Q = self.compute_Q()

        # compute the y vector representing the measured photon counts

        z = l.clone()
        z = torch.bmm(Q,l)
        z = torch.exp(-z)
        z = torch.bmm(S1_S0,z)
        z = torch.poisson(z).to(torch.float32)
        z = torch.bmm(S2,z)
        y = torch.bmm(G,z)

        return y



class SpectralImagingPhysicsModel_MultipleIndependent(SpectralImagingPhysicsModel):
    def __init__(self, spectral_imaging_physics_model_list):

        super(nn.Module, self).__init__()
        # should be a list
        assert isinstance(spectral_imaging_physics_model_list, list), "spectral_imaging_physics_model_list should be a list"
        # should be a list of SpectralImagingPhysicsModel objects
        for spectral_imaging_physics_model in spectral_imaging_physics_model_list:
            assert isinstance(spectral_imaging_physics_model, SpectralImagingPhysicsModel), "spectral_imaging_physics_model_list should be a list of SpectralImagingPhysicsModel objects"
        # store the list of spectral imaging physics models
        self.spectral_imaging_physics_model_list = spectral_imaging_physics_model_list
        self.nMaterial = spectral_imaging_physics_model_list[0].nMaterial
    def compute_Sigma_y(self, l):

        Sigma_y_list = []
        nChannel_list = []
        for spectral_imaging_physics_model in self.spectral_imaging_physics_model_list:
            _Sigma_y = spectral_imaging_physics_model.compute_Sigma_y(l)
            Sigma_y_list.append(_Sigma_y)
            nChannel_list.append(_Sigma_y.shape[1])
        nBatch = _Sigma_y.shape[0]
        nPixel = _Sigma_y.shape[3]

        Sigma_y = torch.zeros(nBatch, sum(nChannel_list), sum(nChannel_list), nPixel).to(_Sigma_y.device)
        for i in range(len(Sigma_y_list)):
            Sigma_y[:,sum(nChannel_list[:i]):sum(nChannel_list[:i+1]),sum(nChannel_list[:i]):sum(nChannel_list[:i+1])] = Sigma_y_list[i]
        
        return Sigma_y

    def compute_grad(self,y, l_hat): 

        grad_list = []
        _nChannel = 0
        for spectral_imaging_physics_model in self.spectral_imaging_physics_model_list:
            _grad = spectral_imaging_physics_model.compute_grad(y[:,_nChannel:_nChannel+spectral_imaging_physics_model.nChannel], l_hat)
            grad_list.append(_grad)
            _nChannel = _nChannel + spectral_imaging_physics_model.nChannel
        nBatch = _grad.shape[0]
        nMaterial = _grad.shape[1]
        nPixel = _grad.shape[2]

        grad = torch.zeros(nBatch, nMaterial, nPixel).to(_grad.device)
        for i in range(len(grad_list)):
            grad = grad + grad_list[i]
        
        return grad

    def compute_F(self, l_background):
        F_list = []
        for spectral_imaging_physics_model in self.spectral_imaging_physics_model_list:
            _F = spectral_imaging_physics_model.compute_F(l_background)
            F_list.append(_F)
        nBatch = _F.shape[0]
        nMaterial = _F.shape[1]
        nPixel = _F.shape[3]

        F = torch.zeros(nBatch, nMaterial, nMaterial, nPixel).to(_F.device)
        for i in range(len(F_list)):
            F = F + F_list[i]

        return F

    def forward(self, l):
        y_list = []
        nChannel_list = []
        for spectral_imaging_physics_model in self.spectral_imaging_physics_model_list:
            _y = spectral_imaging_physics_model(l)
            y_list.append(_y)
            nChannel_list.append(_y.shape[1])
        nBatch = _y.shape[0]
        nPixel = _y.shape[2]

        y = torch.zeros(nBatch, sum(nChannel_list), nPixel).to(_y.device)
        for i in range(len(y_list)):
            y[:,sum(nChannel_list[:i]):sum(nChannel_list[:i+1])] = y_list[i]

        return y

    def forward_with_noise(self, l):
        y_list = []
        nChannel_list = []
        for spectral_imaging_physics_model in self.spectral_imaging_physics_model_list:
            _y = spectral_imaging_physics_model.forward_with_noise(l)
            y_list.append(_y)
            nChannel_list.append(_y.shape[1])
        nBatch = _y.shape[0]
        nPixel = _y.shape[2]

        y = torch.zeros(nBatch, sum(nChannel_list), nPixel).to(_y.device)
        for i in range(len(y_list)):
            y[:,sum(nChannel_list[:i]):sum(nChannel_list[:i+1])] = y_list[i]

        return y
class ModelBasedMaterialDecomposition_MLE(torch.nn.Module):
    def __init__(self, spectral_imaging_physics_model):
        super(ModelBasedMaterialDecomposition_MLE, self).__init__()
        assert isinstance(spectral_imaging_physics_model, SpectralImagingPhysicsModel)

        # store the spectral imaging physics model
        self.spectral_imaging_physics_model = spectral_imaging_physics_model

    def compute_grad(self,y, l_hat):
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   y
        # ---------------------------
        #       Name:
        #           Measurements
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nChannel x nPixel]
        #       Meaning:
        #           The measured detector counts
        #           (not photon counts, includes gain)
        # ---------------------------
        #   l_hat
        # ---------------------------
        #       Name:
        #           Estimated Basis Material Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The line integral of the estimated basis material density (g/mm2)
        # ---------------------------
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   grad
        # ---------------------------
        #       Name:
        #           Fisher Information Matrix
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The Fisher Information Matrix
        #           The inverse of the Cramer-Rao Lower Bound (CRLB) covariance matrix
        #           Relevant for computing the detectability of Maximum Likelihood Estimators (MLE)
        #           It is not the same as Shannon entropy which is commonly called "information" 
        #           The inverse of a covariance matrix is a precision matrix
        #           It describes how precise the measurements are for detecting any combination of materials
        # ---------------------------

        # DELETE ALL THIS IF ITS WORKING FOR A WHILE :) 

        # # check the input
        # assert isinstance(y, torch.Tensor), 'y must be a torch.Tensor'
        # assert y.dtype == torch.float32, 'y must be a torch.float32'
        # assert y.ndim == 3, 'y must be a 3D tensor'
        # nBatch = y.shape[0]
        # nChannel = y.shape[1]
        # nPixel = y.shape[2]
        # # permute the y tensor to be [nBatch x nPixel x nChannel]
        # y = y.permute(0,2,1)
        # # reshape y to be [nBatch*nPixel x nChannel x 1]
        # y = y.reshape(nBatch*nPixel,nChannel,1)

        # # check the other input
        # assert isinstance(l_hat, torch.Tensor), 'l_hat must be a torch.Tensor'
        # assert l_hat.dtype == torch.float32, 'l_hat must be a torch.float32'
        # assert l_hat.ndim == 3, 'l_hat must be a 3D tensor'
        # assert l_hat.shape[0] == nBatch, 'l_hat must have the same number of batches as y'
        # assert l_hat.shape[2] == nPixel, 'l_hat must have the same number of pixels as y'
        # nMaterial = l_hat.shape[1]

        # # compute Q
        # Q = self.spectral_imaging_physics_model.compute_Q()
        # nEnergy = Q.shape[1]
        # nMaterial = Q.shape[2]
        # assert Q.shape == torch.Size([nBatch,nEnergy,nMaterial]), "The shape of Q should be [nBatch x nEnergy x nMaterial]"

        # # compute the D matrix by applying Q to l_hat
        # D = torch.exp(-torch.bmm(Q,l_hat))
        # # at this point, D has shape [nBatch x nEnergy x nPixel]
        # assert D.shape == torch.Size([nBatch,nEnergy,nPixel]), "The shape of D should be [nBatch x nEnergy x nPixel]"
        # # permute D to have shape [nBatch x nPixel x nEnergy]
        # D = D.permute(0,2,1)
        # # at this point, D has shape [nBatch x nPixel x nEnergy]
        # assert D.shape == torch.Size([nBatch,nPixel,nEnergy]), "The shape of D should be [nBatch x nPixel x nEnergy]"
        # # use diag embedding to make D have shape [nBatch x nPixel x nEnergy x nEnergy]
        # D = torch.diag_embed(D)
        # # at this point, D has shape [nBatch x nPixel x nEnergy x nEnergy]
        # assert D.shape == torch.Size([nBatch,nPixel,nEnergy,nEnergy]), "The shape of D should be [nBatch x nPixel x nEnergy x nEnergy]"
        # # combine nBatch and nPixel dimensions
        # D = D.reshape(nBatch*nPixel,nEnergy,nEnergy)
        # # at this point, D has shape [nBatch*nPixel x nEnergy x nEnergy]
        # assert D.shape == torch.Size([nBatch*nPixel,nEnergy,nEnergy]), "The shape of D should be [nBatch*nPixel x nEnergy x nEnergy]"

        
        # # compute the S matrix
        # S = self.spectral_imaging_physics_model.compute_S()
        # # at this point, S has shape [nBatch x nChannel x nEnergy]
        # nChannel = S.shape[1]
        # assert S.shape == torch.Size([nBatch,nChannel,nEnergy]), "The shape of S should be [nBatch x nChannel x nEnergy]"


        # # assume Q is the same for all pixels for now
        # # expand Q to have shape [nBatch x nPixel x nEnergy x nMaterial]
        # Q = Q.unsqueeze(1).expand(nBatch,nPixel,nEnergy,nMaterial)
        # # at this point, Q has shape [nBatch x nPixel x nEnergy x nMaterial]
        # assert Q.shape == torch.Size([nBatch,nPixel,nEnergy,nMaterial]), "The shape of Q should be [nBatch x nPixel x nEnergy x nMaterial]"
        # # combine nBatch and nPixel dimensions
        # Q = Q.reshape(nBatch*nPixel,nEnergy,nMaterial)
        # # at this point, Q has shape [nBatch*nPixel x nEnergy x nMaterial]
        # assert Q.shape == torch.Size([nBatch*nPixel,nEnergy,nMaterial]), "The shape of Q should be [nBatch*nPixel x nEnergy x nMaterial]"


        # # assume S is the same for all pixels for now
        # # expand S to have shape [nBatch x nPixel x nChannel x nEnergy]
        # S = S.unsqueeze(1).expand(nBatch,nPixel,nChannel,nEnergy)
        # # at this point, S has shape [nBatch x nPixel x nChannel x nEnergy]
        # assert S.shape == torch.Size([nBatch,nPixel,nChannel,nEnergy]), "The shape of S should be [nBatch x nPixel x nChannel x nEnergy]"
        # # combine nBatch and nPixel dimensions
        # S = S.reshape(nBatch*nPixel,nChannel,nEnergy)
        # # at this point, S has shape [nBatch*nPixel x nChannel x nEnergy]
        # assert S.shape == torch.Size([nBatch*nPixel,nChannel,nEnergy]), "The shape of S should be [nBatch*nPixel x nChannel x nEnergy]"

        # # compute the measurement covariance
        # Sigma_y = self.spectral_imaging_physics_model.compute_Sigma_y(l_hat)
        # # Sigma_y should have shape [nBatch x nChannel x nChannel x nPixel]
        # assert Sigma_y.shape == torch.Size([nBatch,nChannel,nChannel,nPixel]), "The shape of Sigma_y should be [nBatch x nChannel x nChannel x nPixel]"
        # # permute Sigma_y to have shape [nBatch x nPixel x nChannel x nChannel]
        # Sigma_y = Sigma_y.permute(0,3,1,2)
        # # at this point, Sigma_y has shape [nBatch x nPixel x nChannel x nChannel]
        # assert Sigma_y.shape == torch.Size([nBatch,nPixel,nChannel,nChannel]), "The shape of Sigma_y should be [nBatch x nPixel x nChannel x nChannel]"
        # # combine nBatch and nPixel dimensions
        # Sigma_y = Sigma_y.reshape(nBatch*nPixel,nChannel,nChannel)
        # # at this point, Sigma_y has shape [nBatch*nPixel x nChannel x nChannel]
        # assert Sigma_y.shape == torch.Size([nBatch*nPixel,nChannel,nChannel]), "The shape of Sigma_y should be [nBatch*nPixel x nChannel x nChannel]"
        # # compute the inverse of Sigma_y
        # invSigma_y = torch.linalg.inv(Sigma_y)
        # # at this point, invSigma_y has shape [nBatch*nPixel x nChannel x nChannel]
        # assert invSigma_y.shape == torch.Size([nBatch*nPixel,nChannel,nChannel]), "The shape of invSigma_y should be [nBatch*nPixel x nChannel x nChannel]"


        # # compute y_bar
        # y_bar = self.spectral_imaging_physics_model.forward(l_hat)
        # # y_bar should have shape [nBatch x nChannel x nPixel]
        # assert y_bar.shape == torch.Size([nBatch,nChannel,nPixel]), "The shape of y_bar should be [nBatch x nChannel x nPixel]"
        # # permute y_bar to have shape [nBatch x nPixel x nChannel]
        # y_bar = y_bar.permute(0,2,1)
        # # make it [nBatch*nPixel x nChannel x 1]
        # y_bar = y_bar.reshape(nBatch*nPixel,nChannel,1)
        # assert y_bar.shape == torch.Size([nBatch*nPixel,nChannel,1]), "The shape of y_bar should be [nBatch*nPixel x nChannel x 1]"


        # # Now we need to compute the gradient
        # # the formula for the gradient is:
        # #         grad = torch.matmul(torch.matmul(S_D_Q.T,inv_Simgay),(y - y_bar))
        # #   grad = Q^T * D^T * S^T * invSigma_y * (y - y_bar)
        # # we will do it step by step with batched matrix multiply

        # grad = y - y_bar
        # grad = torch.bmm(invSigma_y, grad)
        # grad = torch.bmm(S.transpose(1,2), grad)
        # grad = torch.bmm(D.transpose(1,2), grad)
        # grad = torch.bmm(Q.transpose(1,2), grad)
        
        # # reshape grad to have shape [nBatch x nPixel x nMaterial]
        # grad = grad.reshape(nBatch,nPixel,nMaterial)
        # # at this point, grad has shape [nBatch x nPixel x nMaterial]
        # assert grad.shape == torch.Size([nBatch,nPixel,nMaterial]), "The shape of grad should be [nBatch x nPixel x nMaterial]"
        # # permute grad to have shape [nBatch x nMaterial x nPixel]
        # grad = grad.permute(0,2,1)
        # # at this point, grad has shape [nBatch x nMaterial x nPixel]
        # assert grad.shape == torch.Size([nBatch,nMaterial,nPixel]), "The shape of grad should be [nBatch x nMaterial x nPixel]"

        # return grad

        return self.spectral_imaging_physics_model.compute_grad(y, l_hat)
    
    def compute_F(self, l_hat):
        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   l_hat
        # ---------------------------
        #       Name:
        #           Basis Material Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The basis material density line integrals
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   F
        # ---------------------------
        #       Name:
        #           Fisher Information Matrix
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nMaterial x nPixel]
        #       Meaning:
        #           Fisher Information Matrix 
        # ---------------------------

        # compute F
        F = self.spectral_imaging_physics_model.compute_F(l_hat)
        return F

    def forward(self, y, l_hat=None, nIter=None):

        # INPUTS:
        #       y                               measurements (photon counts)
        # OUTPUTS:
        #       l_hat                           basis material density line integrals (g/mm2)

        # ---------------------------
        # INPUTS:
        # ---------------------------
        # ---------------------------
        #   y
        # ---------------------------
        #       Name:
        #           Measurements
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nChannel x nPixel]
        #       Meaning:
        #           The measured detector counts
        #           (not photon counts, includes gain)
        # ---------------------------
        #  l_hat (optional)
        # ---------------------------
        #       Name:
        #           Starting Estimate of Basis Material Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Default:
        #           torch.zeros(nBatch, nMaterial, nPixel)
        #       Meaning:
        #           The line integral of the basis material density (g/mm2)
        # ---------------------------
        # nIter (optional)
        # ---------------------------
        #       Name:
        #           Number of Iterations
        #       Type:
        #           int
        #       Default:
        #           10
        #       Meaning:
        #           The number of iterations to run the MLE algorithm
        # ---------------------------
        # ---------------------------
        # OUTPUTS:
        # ---------------------------
        # ---------------------------
        #   l_hat
        # ---------------------------
        #       Name:
        #           Basis Material Density Line Integrals
        #       Type:
        #           torch.Tensor (torch.float32)
        #       Shape:
        #           [nBatch x nMaterial x nPixel]
        #       Meaning:
        #           The line integral of the basis material density (g/mm2)
        # ---------------------------

        # check the input
        assert isinstance(y, torch.Tensor), 'y must be a torch.Tensor'
        assert y.dtype == torch.float32, 'y must be a torch.float32'
        # 3 dimensions: nBatch x nChannel x nPixel
        assert len(y.shape) == 3, 'y must be a 3 dimensional tensor'
        nBatch = y.shape[0]
        nChannel = y.shape[1]
        nPixel = y.shape[2]

        # Q = self.spectral_imaging_physics_model.compute_Q()
        # # 3 dimensions: nBatch x nEnergy x nMaterial
        # assert len(Q.shape) == 3, 'Q must be a 3 dimensional tensor'
        # nEnergy = Q.shape[1]
        # nMaterial = Q.shape[2]
        nMaterial = self.spectral_imaging_physics_model.nMaterial

        # check the l_hat input (optional)
        if l_hat is None:
            l_hat = torch.zeros([nBatch,nMaterial,nPixel], dtype=torch.float32, device=y.device)
        assert isinstance(l_hat, torch.Tensor), 'l_hat must be a torch.Tensor'
        assert l_hat.dtype == torch.float32, 'l_hat must be a torch.float32'
        # 3 dimensions: nBatch x nMaterial x nPixel
        assert len(l_hat.shape) == 3, 'l_hat must be a 3 dimensional tensor'
        assert l_hat.shape == torch.Size([nBatch,nMaterial,nPixel]), 'l_hat must be a [nBatch x nMaterial x nPixel] tensor'

        # check the nIter input (optional)
        if nIter is None:
            nIter = 10
        assert isinstance(nIter, int), 'nIter must be an int'

        for iIter in range(nIter):

            # compute the expected measurements
            y_bar = self.spectral_imaging_physics_model.forward(l_hat)

            # compute the Fisher Information Matrix
            F = self.compute_F(l_hat)
            # F should be shape [nBatch x nMaterial x nMaterial x nPixel]
            # permute to [nBatch x nPixel x nMaterial x nMaterial]
            F = F.permute(0,3,1,2)
            # reshape to [nBatch*nPixel x nMaterial x nMaterial]
            F = F.reshape(nBatch*nPixel,nMaterial,nMaterial)
            # at this point, F should have shape [nBatch*nPixel x nMaterial x nMaterial]
            assert F.shape == torch.Size([nBatch*nPixel,nMaterial,nMaterial]), 'F must be a [nBatch*nPixel x nMaterial x nMaterial] tensor'

            # get the diagonal elements of F used for proximal regularization
            # extract the diagonal elements of F
            F_diag = torch.diagonal(F, dim1=1, dim2=2)
            # add a constant which is the mean of the diagonal elements of F
            # the shape should be [nBatch*nPixel x nMaterial]
            F_diag = F_diag + 0.001*torch.mean(F_diag, dim=1, keepdim=True)
            # at this point, F_diag should have shape [nBatch*nPixel x nMaterial]
            # make it a diagonal matrix
            F_diag = torch.diag_embed(F_diag)
            # the proximal regularization term is 1% of the diagonal elements of F
            # note, this does not affect the loss function. 
            # the gradient is for the original loss function. 
            R = 0.001*F_diag
            # at this point, R should have shape [nBatch*nPixel x nMaterial x nMaterial]
            assert R.shape == torch.Size([nBatch*nPixel,nMaterial,nMaterial]), 'R must be a [nBatch*nPixel x nMaterial x nMaterial] tensor'

            # compute the gradient of the loss function
            grad = self.compute_grad(y, l_hat)
            # grad should be shape [nBatch x nMaterial x nPixel]
            # permute to [nBatch x nPixel x nMaterial]
            grad = grad.permute(0,2,1)
            # reshape to [nBatch*nPixel x nMaterial x 1]
            grad = grad.reshape(nBatch*nPixel,nMaterial,1)

            # compute the newtons method update
            delta_l_hat = - torch.bmm(torch.linalg.inv(F+R),grad)
            # reshape to [nBatch x nPixel x nMaterial]
            delta_l_hat = delta_l_hat.reshape(nBatch,nPixel,nMaterial)
            # permute to [nBatch x nMaterial x nPixel]
            delta_l_hat = delta_l_hat.permute(0,2,1)
    
            l_hat = l_hat + delta_l_hat

            
        return l_hat



def compute_separability(F):
    # ---------------------------
    # INPUTS:
    # ---------------------------
    # ---------------------------
    #   F
    # ---------------------------
    #       Name:
    #           Fisher Information Matrix
    #       Type:
    #           torch.Tensor (torch.float32)
    #       Shape:
    #           [nBatch x nMaterial x nMaterial x nPixel]
    #       Meaning:
    #           The Fisher Information Matrix
    # ---------------------------
    # ---------------------------
    # OUTPUTS:
    # ---------------------------
    # ---------------------------
    #   separability
    # ---------------------------
    #       Name:
    #           Separability Index
    #       Type:
    #           torch.Tensor (torch.float32)
    #       Shape:
    #           [nBatch x nPixel]
    #       Meaning:
    #           The separability index
    # ---------------------------

    # check the input
    assert isinstance(F, torch.Tensor), 'F must be a torch.Tensor'
    assert F.dtype == torch.float32, 'F must be a torch.float32'
    # 4 dimensions: nBatch x nMaterial x nMaterial x nPixel
    assert len(F.shape) == 4, 'F must be a 4 dimensional tensor'
    nBatch = F.shape[0]
    nMaterial = F.shape[1]
    nPixel = F.shape[3]
    assert F.shape == torch.Size([nBatch,nMaterial,nMaterial,nPixel]), 'F must be a [nBatch x nMaterial x nMaterial x nPixel] tensor'
    # permute to [nBatch x nPixel x nMaterial x nMaterial]
    F = F.permute(0,3,1,2)
    # reshape to [nBatch*nPixel x nMaterial x nMaterial]
    F = F.reshape(nBatch*nPixel,nMaterial,nMaterial)
    # at this point, F should have shape [nBatch*nPixel x nMaterial x nMaterial]
    assert F.shape == torch.Size([nBatch*nPixel,nMaterial,nMaterial]), 'F must be a [nBatch*nPixel x nMaterial x nMaterial] tensor'
    # extract the diagonal elements of F
    F_diag = torch.diagonal(F, dim1=1, dim2=2)
    # at this point, F_diag should have shape [nBatch*nPixel x nMaterial]
    # make it a diagonal matrix
    inv_sqrtm_F_diag = torch.diag_embed(1/torch.sqrt(F_diag))
    # normalize the Fisher Information Matrix symmetrically
    F_norm = torch.bmm(torch.bmm(inv_sqrtm_F_diag,F),inv_sqrtm_F_diag)
    # separability is one over the square root of the condition number of the normalized Fisher Information Matrix
    separability = 1.0/torch.sqrt(torch.linalg.cond(F_norm))
    # reshape to [nBatch x nPixel]
    separability = separability.reshape(nBatch,nPixel)
    # at this point, separability should have shape [nBatch x nPixel]
    assert separability.shape == torch.Size([nBatch,nPixel]), 'separability must be a [nBatch x nPixel] tensor'

    return separability


