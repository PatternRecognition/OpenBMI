function out= proc_dBaverage(epo, varargin)
%PROC_DBAVERAGE - Classwise calculated averages for dB-scaled features
%
%This functions is exactly used as proc_average. It should be used
%for dB-scaled features (e.g. output of proc_power2dB; or
%proc_spectrum in the default setting 'scaling', 'dB').

out= copy_struct(epo, 'not','x','yUnit');
%% scale back
out.x= 10.^(epo.x/10);
%% average
out= proc_average(out, varargin{:});
%% re-convert to dB
out.x= 10*log10(out.x);
out.yUnit= 'dB';
