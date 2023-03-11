
import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg') # dont use X11
import pydicom

# plt.style.use('dark_background')

import torch

import os

import numpy as np
import time

from xml.dom import minidom

import types

from scipy.ndimage import binary_fill_holes, binary_dilation, gaussian_filter






def load_lidc_images(applyMask=False):
    # load the numpy file
    lidc_images = np.load('/home/matt/Research/20221207_lidc/data/lidc_nodule_database/images.npy')
    if applyMask:
        lidc_masks = np.load('/home/matt/Research/20221207_lidc/data/lidc_nodule_database/masks.npy')
        lidc_images = lidc_images * lidc_masks
    lidc_images = lidc_images.reshape(-1, 1, 64, 64)
    lidc_images = lidc_images - 1000 # Hounsfield units
    return lidc_images









lidc_path = '/home/matt/Research/LIDC-IDRI/manifest-1600709154662/LIDC-IDRI'


# file = minidom.parse('/home/matt/Research/LIDC-IDRI/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/069.xml')
# nodule = file.getElementsByTagName('readingSession')[0].getElementsByTagName('unblindedReadNodule')[0]
# nodule_id = nodule.getElementsByTagName('noduleID')[0].firstChild.nodeValue
# subtlety = nodule.getElementsByTagName('subtlety')[0].firstChild.nodeValue
# internalStructure = nodule.getElementsByTagName('internalStructure')[0].firstChild.nodeValue
# calcification = nodule.getElementsByTagName('calcification')[0].firstChild.nodeValue
# sphericity = nodule.getElementsByTagName('sphericity')[0].firstChild.nodeValue
# margin = nodule.getElementsByTagName('margin')[0].firstChild.nodeValue
# lobulation = nodule.getElementsByTagName('lobulation')[0].firstChild.nodeValue
# spiculation = nodule.getElementsByTagName('spiculation')[0].firstChild.nodeValue
# texture = nodule.getElementsByTagName('texture')[0].firstChild.nodeValue
# malignancy = nodule.getElementsByTagName('malignancy')[0].firstChild.nodeValue

class nodule():
    def __init__(self, nodule_id, subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture, malignancy,roi):
        self.nodule_id = nodule_id
        self.subtlety = subtlety
        self.internalStructure = internalStructure
        self.calcification = calcification
        self.sphericity = sphericity
        self.margin = margin
        self.lobulation = lobulation
        self.spiculation = spiculation
        self.texture = texture
        self.malignancy = malignancy
        self.roi = roi

    def __repr__(self):
        return "Nodule #{}: subtlety={}, internalStructure={}, calcification={}, sphericity={}, margin={}, lobulation={}, spiculation={}, texture={}, malignancy={}".format(self.nodule_id, self.subtlety, self.internalStructure, self.calcification, self.sphericity, self.margin, self.lobulation, self.spiculation, self.texture, self.malignancy)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.nodule_id == other.nodule_id

    def __hash__(self):
        return hash(self.nodule_id)


def load_lidc_xml(iPatient=None, iScan=None, verbose=False):

    assert iPatient is not None, "Must specify a patient number"
    # assert iScan is not None, "Must specify a scan number"

    assert iPatient <= 1000, "Patient number must be less than 1000"
    assert iPatient > 0, "Patient number must be greater than 0"
    # for some reason 238 and 585 are missing. so we map patient 1001 and 1002
    if iPatient == 238:
        iPatient = 1001
    if iPatient == 585:
        iPatient = 1002
    
    patient_path = os.path.join(lidc_path, 'LIDC-IDRI-{:04d}'.format(iPatient))
    
    possible_scan_paths = [os.path.join(os.path.join(patient_path,scan_path), os.listdir(os.path.join(patient_path,scan_path))[0]) for scan_path in os.listdir(patient_path)]
    number_of_dcm_files_in_scan_paths = [len(os.listdir(os.path.join(patient_path, scan_path))) for scan_path in possible_scan_paths]
    sorted_scan_paths = [x for _,x in sorted(zip(number_of_dcm_files_in_scan_paths, possible_scan_paths))]

    if iScan is None:
        iScan = len(sorted_scan_paths) - 1
        if verbose:
            print("No scan number specified, loading scan #{} (the largest one)".format(iScan))
    scan_path = sorted_scan_paths[iScan]

    # get the list of xml files in the scan path ending in the file extension *.xml
    xml_files = [os.path.join(scan_path, xml_file) for xml_file in os.listdir(scan_path) if xml_file.endswith('.xml')]
    # assume we only have one xml file per scan
    if not xml_files:
        return None
    xml_file = xml_files[0]

    # parse the xml file
    _file = minidom.parse(xml_file)
    _readingSessions = _file.getElementsByTagName('readingSession')

    nodule_list = []
    for readingSession in _readingSessions:
        _unblindedReadNodules = readingSession.getElementsByTagName('unblindedReadNodule')
        for _unblindedReadNodule in _unblindedReadNodules:
            nodule_id = _unblindedReadNodule.getElementsByTagName('noduleID')[0].firstChild.nodeValue
            if _unblindedReadNodule.getElementsByTagName('characteristics'):
                try:
                    subtlety = int(_unblindedReadNodule.getElementsByTagName('characteristics')[0].getElementsByTagName('subtlety')[0].firstChild.nodeValue)
                    internalStructure = int(_unblindedReadNodule.getElementsByTagName('characteristics')[0].getElementsByTagName('internalStructure')[0].firstChild.nodeValue)
                    calcification = int(_unblindedReadNodule.getElementsByTagName('characteristics')[0].getElementsByTagName('calcification')[0].firstChild.nodeValue)
                    sphericity = int(_unblindedReadNodule.getElementsByTagName('characteristics')[0].getElementsByTagName('sphericity')[0].firstChild.nodeValue)
                    margin = int(_unblindedReadNodule.getElementsByTagName('characteristics')[0].getElementsByTagName('margin')[0].firstChild.nodeValue)
                    lobulation = int(_unblindedReadNodule.getElementsByTagName('characteristics')[0].getElementsByTagName('lobulation')[0].firstChild.nodeValue)
                    spiculation = int(_unblindedReadNodule.getElementsByTagName('characteristics')[0].getElementsByTagName('spiculation')[0].firstChild.nodeValue)
                    texture = int(_unblindedReadNodule.getElementsByTagName('characteristics')[0].getElementsByTagName('texture')[0].firstChild.nodeValue)
                    malignancy = int(_unblindedReadNodule.getElementsByTagName('malignancy')[0].firstChild.nodeValue)
                except:
                    print("Error parsing characteristics for nodule {}".format(nodule_id))
                    return None
            else:
                subtlety = None
                internalStructure = None
                calcification = None
                sphericity = None
                margin = None
                lobulation = None
                spiculation = None
                texture = None
                malignancy = None
            roi = []
            if _unblindedReadNodule.getElementsByTagName('roi'):
                for _roi in _unblindedReadNodule.getElementsByTagName('roi'):
                    imageZposition = float(_roi.getElementsByTagName('imageZposition')[0].firstChild.nodeValue)
                    xCoord = []
                    yCoord = []
                    for edgeMap in _roi.getElementsByTagName('edgeMap'):
                        xCoord.append(int(edgeMap.getElementsByTagName('xCoord')[0].firstChild.nodeValue))
                        yCoord.append(int(edgeMap.getElementsByTagName('yCoord')[0].firstChild.nodeValue))
                    roi.append((imageZposition, xCoord, yCoord))
            nodule_list.append(nodule(nodule_id, subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture, malignancy,roi))
    
    if not nodule_list:
        return None
    
    return nodule_list

node_list = load_lidc_xml(iPatient=1, iScan=1, verbose=True)





def get_dicom_path(iPatient=None, iScan=None, iSlice=None, verbose=False):

    if iPatient is None:
        iPatient = np.random.randint(1, 999)
        if verbose:
            print("No patient number specified, randomly loading patient #{}".format(iPatient))

    assert iPatient <= 1000, "Patient number must be less than 1000"
    assert iPatient > 0, "Patient number must be greater than 0"

    # for some reason 238 and 585 are missing. so we map patient 1001 and 1002
    if iPatient == 238:
        iPatient = 1001
    if iPatient == 585:
        iPatient = 1002

    patient_path = os.path.join(lidc_path, 'LIDC-IDRI-{:04d}'.format(iPatient))
    
    possible_scan_paths = [os.path.join(os.path.join(patient_path,scan_path), os.listdir(os.path.join(patient_path,scan_path))[0]) for scan_path in os.listdir(patient_path)]
    number_of_dcm_files_in_scan_paths = [len(os.listdir(os.path.join(patient_path, scan_path))) for scan_path in possible_scan_paths]
    sorted_scan_paths = [x for _,x in sorted(zip(number_of_dcm_files_in_scan_paths, possible_scan_paths))]

    if iScan is None:
        iScan = len(sorted_scan_paths) - 1
        if verbose:
            print("No scan number specified, loading scan #{} (the largest one)".format(iScan))
    scan_path = sorted_scan_paths[iScan]

    # get the list of dcm files in the scan path ending in the file extension *.dcm
    dcm_files = [os.path.join(scan_path, dcm_file) for dcm_file in os.listdir(scan_path) if dcm_file.endswith('.dcm')]
    dcm_files = sorted(dcm_files)

    if iSlice is None:
        iSlice = np.random.randint(0, len(dcm_files))
        if verbose:
            print("No slice number specified, randomly loading slice #{}".format(iSlice))
    if iSlice >= len(dcm_files):
        iSlice = len(dcm_files) - 1
        if verbose:
            print("Slice number specified is too large, loading last slice #{}".format(iSlice))
    dcm_path = os.path.join(scan_path, dcm_files[iSlice])

    return (iPatient, iScan, iSlice, patient_path, scan_path, dcm_path)


def load_dicom(dcm_path):

    ds = pydicom.dcmread(dcm_path)
    
    if not hasattr(ds,'PixelSpacing'):
        return None

    if not hasattr(ds,'SliceLocation'):
        return None
    
    pixel_spacing = ds.PixelSpacing
    Nx = ds.pixel_array.shape[1]
    Ny = ds.pixel_array.shape[0]
    
    # use torch to interpolate the image to the xvRegister,yvRegister grid
    img = torch.from_numpy(ds.pixel_array.astype(float))
    # img[img<0] = 0
    # img[img>3000] = 3000
    img = img.type(torch.float32)

    # use the dicom metadata to shift and scale the image values
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    img = img * slope + intercept
    img = img + 1000

    return img, pixel_spacing, ds


def register_image(img, pixel_spacing):

    ps_target = 1.0
    Ny = 512
    Nx = 512
    
    # use torch to set up a meshgrid for the image
    x = torch.linspace(0, Nx-1, Nx)
    y = torch.linspace(0, Ny-1, Ny)
    xvRegister, yvRegister = torch.meshgrid(x, y)
    
    # use torch to set up a meshgrid for the image
    x = torch.linspace(0, Nx-1, Nx)
    y = torch.linspace(0, Ny-1, Ny)
    xv, yv = torch.meshgrid(x, y)

    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = torch.nn.functional.interpolate(img, scale_factor=pixel_spacing[0]/ps_target, mode='bilinear', align_corners=False)
    img = img.squeeze(0)
    img = img.squeeze(0)
    
    # image the image is smaller than 512x512, pad it with zeros using torch
    if img.shape[0] < 512:
        img = torch.nn.functional.pad(img, ((512-img.shape[0])//2, (513-img.shape[0])//2, (512-img.shape[1])//2, (513-img.shape[1])//2), mode='constant', value=0)

    # image the image is larger than 512x512, crop it using torch
    if img.shape[0] > 512:
        img = img[(img.shape[0]-512)//2:(img.shape[0]+512)//2, (img.shape[1]-512)//2:(img.shape[1]+512)//2]










def extract_lidc_lesion_dataset():
        
    inds = np.arange(1000)+1
    # np.random.shuffle(inds)

    max_nodules = 100000

    nodule_image_stack = np.zeros([max_nodules, 64, 64])
    nodule_mask_stack = np.zeros([max_nodules, 64, 64])
    nodule_subtlety_stack = np.zeros([max_nodules])
    nodule_internalStructure_stack = np.zeros([max_nodules])
    nodule_calcification_stack = np.zeros([max_nodules])
    nodule_sphericity_stack = np.zeros([max_nodules])
    nodule_margin_stack = np.zeros([max_nodules])
    nodule_lobulation_stack = np.zeros([max_nodules])
    nodule_spiculation_stack = np.zeros([max_nodules])
    nodule_texture_stack = np.zeros([max_nodules])
    nodule_malignancy_stack = np.zeros([max_nodules])

    nodule_lidc_id_stack = np.zeros([max_nodules], dtype=object)
    nodule_location_col_stack = np.zeros([max_nodules])
    nodule_location_row_stack = np.zeros([max_nodules])
    nodule_location_z_stack = np.zeros([max_nodules])

    iNodule = 0

    for iPatient in range(1000):

        if iNodule >= max_nodules:
            break

        nodule_list = load_lidc_xml(iPatient=inds[iPatient], iScan=None, verbose=True)

        if nodule_list is None:
            continue

        roi_list = []
        another_nodule_list = []
        for nodule in nodule_list:
            for roi in nodule.roi:
                roi_list.append(roi)
                another_nodule_list.append(nodule)
        
        sliceLocations = torch.tensor([roi[0] for roi in roi_list])

        nodule_id_list = []

        for iRoi in range(len(roi_list)):

            if iNodule >= max_nodules:
                break
            
            # if we have already seen this nodule id, skip it
            if another_nodule_list[iRoi].nodule_id+'_'+str(sliceLocations[iRoi]) in nodule_id_list:
                continue
            else:
                nodule_id_list.append(another_nodule_list[iRoi].nodule_id+'_'+str(sliceLocations[iRoi]))

            if len(roi_list[iRoi][1])<3:
                continue

            if another_nodule_list[iRoi].subtlety is None:
                continue

            current_nodule = another_nodule_list[iRoi]

            nodule_sliceLocation = sliceLocations[iRoi]

            (jPatient, jScan, jSlice, patient_path, scan_path, dcm_path) = get_dicom_path(iPatient=inds[iPatient], iScan=None, iSlice=0, verbose=False)

            img, pixel_spacing, ds = load_dicom(dcm_path)

            sliceLocation = ds.SliceLocation
            sliceThickness = ds.SliceThickness

            nodule_sliceIndex = torch.floor((sliceLocation - nodule_sliceLocation)/sliceThickness).type(torch.int64)
            if nodule_sliceIndex < 0:
                nodule_sliceIndex = torch.floor((nodule_sliceLocation - sliceLocation)/sliceThickness).type(torch.int64)

            (jPatient, jScan, jSlice, patient_path, scan_path, dcm_path) = get_dicom_path(iPatient=inds[iPatient], iScan=None, iSlice=nodule_sliceIndex, verbose=False)
            img, pixel_spacing, ds = load_dicom(dcm_path)

            if img is None:
                continue
            if torch.mean(img)<50:
                continue
            if torch.abs(ds.SliceLocation - nodule_sliceLocation) > 5:
                continue

            # clear the axes
            img_mask = img*0
            img_mask[roi_list[iRoi][2], roi_list[iRoi][1]] = 1
            img_mask = binary_fill_holes(img_mask)
            img_mask = binary_dilation(img_mask)
            img_mask = binary_dilation(img_mask)
            # img_mask = binary_dilation(img_mask)
            img_mask = img_mask.astype(float)
            img_mask = gaussian_filter(img_mask, sigma=1.0)
            # img_mask[img_mask2] = 1
            # img_mask[roi_list[iRoi][2], roi_list[iRoi][1]] = 3000
            
            # im.set_clim([0,1])
            # ax.plot(roi_list[iRoi][1], roi_list[iRoi][2], 'r.')

            x_mean = np.floor(np.mean(roi_list[iRoi][1]))
            y_mean = np.floor(np.mean(roi_list[iRoi][2]))

            xl = np.floor(x_mean-32)
            xu =  xl + 64
            yl = np.floor(y_mean-32)
            yu = yl + 64

            xl = xl.astype(int)
            xu = xu.astype(int)
            yl = yl.astype(int)
            yu = yu.astype(int)

            nodule_img = img[yl:yu, xl:xu]
            nodule_mask = img_mask[yl:yu, xl:xu]

            nodule_image_stack[iNodule, :, :] = nodule_img
            nodule_mask_stack[iNodule, :, :] = nodule_mask
            nodule_calcification_stack[iNodule] = current_nodule.calcification
            nodule_internalStructure_stack[iNodule] = current_nodule.internalStructure
            nodule_subtlety_stack[iNodule] = current_nodule.subtlety
            nodule_sphericity_stack[iNodule] = current_nodule.sphericity
            nodule_margin_stack[iNodule] = current_nodule.margin
            nodule_lobulation_stack[iNodule] = current_nodule.lobulation
            nodule_spiculation_stack[iNodule] = current_nodule.spiculation
            nodule_texture_stack[iNodule] = current_nodule.texture
            nodule_malignancy_stack[iNodule] = current_nodule.malignancy

            nodule_lidc_id_stack[iNodule] = iPatient
            nodule_location_col_stack[iNodule] = x_mean
            nodule_location_row_stack[iNodule] = y_mean
            nodule_location_z_stack[iNodule] = nodule_sliceLocation

            iNodule += 1
            
            print('iPatient ', iPatient, ' number of nodules: ', len(nodule_list), ' iNodule: ', iNodule)



def load_lidc_axial_images(iPatient=None, iSlice=None, verbose=False):

    if iPatient is None:    
        iPatient = np.random.randint(0, 1000) + 1

    (jPatient, jScan, jSlice, patient_path, scan_path, dcm_path) = get_dicom_path(iPatient=iPatient, iScan=None, iSlice=0, verbose=verbose)
    
    if iSlice is None:
        np.random.randint(0,len(os.listdir(scan_path)))

    (jPatient, jScan, jSlice, patient_path, scan_path, dcm_path) = get_dicom_path(iPatient=iPatient, iScan=None, iSlice=iSlice, verbose=verbose)

    img, pixel_spacing, ds = load_dicom(dcm_path)

    img = img.reshape(-1, 1, 512,512)
    img = img - 1000 # Hounsfield units

    return img



