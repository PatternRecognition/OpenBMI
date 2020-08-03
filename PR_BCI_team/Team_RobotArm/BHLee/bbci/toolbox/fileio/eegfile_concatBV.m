function [varargout]= eegfile_concatBV(file_list, varargin)
% EEGFILE_CONCATBV - concatenate files which are stored in BrainVision format
%
% Synopsis:
%   [DAT, MRK, MNT]= eegfile_concatBV(FILE_LIST, 'Property, 'Value', ...)
%
% Arguments:
%   FILE_LIST: list of file names (no extension)
%
% Returns:
%   DAT: structure of continuous or epoched signals
%   MRK: marker structure
%   MNT: electrode montage structure
%
% Properties:
%   are passed to eegfile_loadBV
%
% Description:
%   This function is called by eegfile_loadBV in case the file name argument
%   is a cell array of file names. Typically there is no need to call this 
%   function directly.
%
% See also: eegfile_*
%

if ~iscell(file_list),
  file_list= {file_list};
end

T= zeros(1, length(file_list));

for ii= 1:length(file_list),
  [cnt, mrk, hdr]= eegfile_loadBV(file_list{ii}, varargin{:});
  T(ii)= size(cnt.x,1);
  if ii==1,
    ccnt= cnt;
    cmrk= mrk;
  else
    if ~isequal(cnt.clab, ccnt.clab),
      warning(['inconsistent clab structure will be repaired ' ...
               'by using the intersection']); 
      commonclab= intersect(cnt.clab, ccnt.clab);
      cnt= proc_selectChannels(cnt, commonclab{:});
      ccnt= proc_selectChannels(ccnt, commonclab{:});
    end
    if ~isequal(cnt.fs, ccnt.fs)
        error('inconsistent sampling rate'); 
    end
    ccnt.x= cat(1, ccnt.x, cnt.x);
    if length(cmrk)>1,
      shift= sum(TT(1:ii-1));
      for ii= 1:length(mrk),
        mrk(ii).pos= mrk(ii).pos + shift;
      end
      cmrk= bvmrk_appendMarkers(cmrk, mrk);
    else
      mrk.pos= mrk.pos + sum(T(1:ii-1));
      cmrk= mrk_mergeMarkers(cmrk, mrk);
    end
  end
end

ccnt.T= T;
if length(file_list)>1,
  ccnt.title= [ccnt.title ' et al.'];
  ccnt.file= strcat(fileparts(ccnt.file), file_list);
end

varargout= cell(1, nargout);
varargout{1}= ccnt;
if nargout>1,
  varargout{2}= cmrk;
  if nargout>2,
    varargout{3}= hdr;
  end
end
