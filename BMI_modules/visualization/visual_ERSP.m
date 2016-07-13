function [ ersp ] = visual_ERSP( data, varargin )
% ERSP : "Event-related spectral pergurbation" 
% Measuring the average dynamic changes in amplitude of the broad band
% EEG Frequency band
%
% Synopsis:
%  [dat] = proc_ERSP(data , <OPT>)
%
% Arguments:
%   data: Data structrue (ex) Epoched data structure
%   <OPT> : 
%      .Channel - Selecting the interested channel in Time-Frequency domain
%                 (e.g. {'Channel', {'C4'}})
%      .Interval - Selecting the interested time intervals
%                 (e.g. {'Interval' , '[-2000 3000]'})
%
% Return:
%    data:  Epoched data structure
%
% See also:
%    opt_cellToStruct , visual_timef
%
% Reference:
%         1. C. Brunner, A. Delorme, and S. Makeig, "EEGLAB?An Open Source 
%          Matlab Toolbox for Electrophysiological Research," Biomedical Engineering/Biomedizinische Technik, 2013, pp.1-2.
%         2. EEGLAB tutorial (http://sccn.ucsd.edu/eeglab/)
%          Author: Sigurd Enghoff, Arnaud Delorme & Scott Makeig
%          CNL / Salk Institute 1998- | SCCN/INC, UCSD 2002-
%
%         We used EEGLAB open source toolbox code related in ERSP (timef.m)  
%
%  
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
%
%%
dat = data;
opt = opt_cellToStruct(varargin{:});
chinx = find(strcmp(dat.chan,opt.Channel)==1);
eeg = [];

if ndims(dat.x) == 2
    if ~isfield(opt,'Interval')
        warning('OpenBMI: please input the data intervals');return;
    end
    dat = prep_segmentation(dat, {'interval',opt.Interval});
    dat.x = dat.x(:,:,chinx);
    eeg = reshape(dat.x , [size(dat.x,1)*size(dat.x,2) 1])';
elseif ndims(dat.x) == 3
    dat.x = dat.x(:,:,chinx);
    eeg = reshape(dat.x , [size(dat.x,1)*size(dat.x,2) 1])';
end

figure; 
ersp = visual_timef(eeg, size(dat.x,1), [dat.ival(1) dat.ival(end)], dat.fs, [3 0.5], 'plotersp','on','plotitc','off');


