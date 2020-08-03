function [Cnt, Mrk]= loadGenericEEG_int(file_name, varargin)
% LOADGENERICEEG - Load EEG data (and markers) which are stored in
%   generic data format (BrainVision Format)
%
% Synopsis:
%   CNT= loadGenericEEG(FILE_NAME)
%   CNT= loadGenericEEG(FILE_NAME, PROPERTY VALUE LIST)
%   [CNT, MRK]= loadGenericEEG(FILE_NAME)
%   [CNT, MRK]= loadGenericEEG(FILE_NAME, PROPERTY VALUE LIST)
%
% Arguments:
%   FILE_NAME: string, or cell array of strings. File names that are not
%    absolute, are taken relative to EEG_RAW_DIR (global variable)
% optional fields
%   .clab: name of the channels to be loaded, default [] means all
%   .fs: sampling rate, default 'raw' means original sampling rate
%   .from: start [msec] from which loading should start, default [] means 0
%   .maxlen: maximum length [msec] to be loaded, default [] means all
%   
% Returns:
%   CNT - data structure of continuous EEG data
%   MRK - data structure of EEG markers
%
% Description:
%   Watch out: in contrast to READGENERICEEG, this function uses the
%   original sampling rate as default.
%  
% See also:
%   readGenericEEG, readGenericMarkers (to be written)

% benjamin.blankertz@first.fhg.de, Nov 2004

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'clab', [], ...
                  'fs', 'raw', ...
                  'from', 0, ...
                  'maxlen', []);

if ~iscell(file_name),
  file_list= {file_name};
else
  file_list= file_name;  
end

mrk= [];
for ff= 1:length(file_list),
  file_name= file_list{ff};
  cnt= readGenericEEG_int(file_name, opt.clab, opt.fs, opt.from, opt.maxlen);
  if nargout>1,
    %% in future this should be replaced by 
    %% mrk= readGenericMarker, a function which reads *all* markers.
    mrk= readMarkerTable(file_name, opt.fs);
    mrk.pos= mrk.pos - round(opt.from/1000*opt.fs);
    iValid= find(mrk.pos>0 & mrk.pos<size(cnt.x,1));
    mrk= mrk_selectEvents(mrk, iValid);
  end
  if ff==1,
    Cnt= cnt;
    Mrk= mrk;
  else
    [Cnt, Mrk]= proc_appendCnt(Cnt, cnt, Mrk, mrk);
  end
end
