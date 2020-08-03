%This folder contains the so-called CALIBRATE functions, to be called
%by bbci_calibrate. These functions receive as in put the calibration 
%data and the BBCI structure, which holds specific parameters for the
%calibration procedure in the field BBCI.calibrate. The output is
%and updated BBCI structure which has all the necessary information
%for online operation, i.e., for bbci_apply, see bbci_apply_strctures.
%
%The CALIBRATE functions have the following format:
%
% --- --- --- ---
%BBCI_CALIBRATE_XYZ - Calibrate online system for paradigm XYZ
%
%Synopsis:
% [BBCI, DATA]= bbci_calibate_XYZ(BBCI, DATA)
% 
%Arguments:
%  BBCI - The field BBCI.data holds the calibration data and the field
%     'calibrate' holds parameters specific to calibration XYZ.
%  DATA - Hold the calibration data in fields 'cnt', 'mrk', and 'mnt'.
%     Furthermore, DATA.isnew indicates whether calibration data is
%     new (loaded for the first time or reloaded) or whether it is the
%     same as before. In the latter case, some steps of calibration
%     might be omitted (subject to changes in BBCI.calibrate.settings).
%     In order to check, what settings have been changed since the last
%     run, DATA.previous_settings hold the settings of the previous
%     calibration run.
%  
%Output:
%  BBCI - Updated BBCI structure in which all necessary fields for
%     online operation are set, see bbci_apply_structures.
%  DATA - Updated DATA structure, with the result of selections being
%     stored in DATA.result.
%     Furthermore, DATA.figure_handles should hold the handles of all
%     figures that should be stored by bbci_save. If this field is not
%     defined, bbci_save will save all Matlab figure (if saving figures
%     is requested by BBCI.calibrate.save.figures==1).
%
%To get a description on the structures 'BBCI' and 'DATA', type
%help bbci_calibrate_structures
%
%Conventions:
% The CALIBRATE functions should *only* read the (sub)field
% bbci.calibrate.settings. However, this field should *not* be modified.
% (It is debateble, whether default values for unspecified parameters
% should be filled in.) Selection of values for parameters which are
% unspecified by the user (or specified as 'auto') should *not* be stored
% in bbci.calibrate.settings, but in data.result under the save field name.
%
% --- --- --- ---
%
%List of CALIBRATE functions (prefix bbci_calibrate_ is left out)
% - ERP_Speller: Setup for the online system to perform classification
%      for an ERP Speller in the stardard format.
% - csp: Setup for classifying SMR Modulations with CSP filters and
%      log band-power features
% - csp_plus_lap: Additionally to optimized CSP filters some Laplacian
%      channels are selected and used for classification. These are meant
%      to be reselected during supervised adaptation with
%      bbci_adaptation_csp_plus_lap.
