function [varargout]= eegfile_concatMatlab(file_list, varargin)
% EEGFILE_CONCATMATLAB - concatenate EEG Matlab structures
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

opt= propertylist2struct(varargin{:});
if nargout~=length(opt.vars),
  warning('number of output arguments does not match with requested vars');
end

iDat= find(ismember(opt.vars, {'dat','cnt','epo'}));
iMrk= find(strcmp(opt.vars, 'mrk'));
%iNoCat= setdiff(1:length(opt.vars), [iDat iMrk]);
iMrkBV= find(strcmp(opt.vars, 'mrk_orig'));
iNfo= find(strcmp(opt.vars, 'nfo'));
if isempty(iNfo),
  opt.vars= cat(2, opt.vars, {'nfo'});
  iNfo= length(opt.vars);
end
varargcat= cell(1,length(opt.vars));

if ~iscell(file_list),
  file_list= {file_list};
end

N= NaN*zeros(1, length(file_list));
T= zeros(1, length(file_list));

for ii= 1:length(file_list),
  [varargcat{:}]= eegfile_loadMatlab(file_list{ii}, opt);
  T(ii)= varargcat{iNfo}.T;
  N(ii)= max(varargcat{iNfo}.nEpochs, varargcat{iNfo}.nEvents);
  if ii==1,
    varargout= varargcat(1:nargout);
  else
%%The following code does not work, because isequal(NaN,NaN)= 0.
%    for kk= iNoCat,  %% check variables that should be constant
%      if ~isequal(varargout(kk), varargcat(kk)),
%        warning(sprintf('inconsistency regarding variable <%s>', ...
%                        opt.vars{kk}));
%      end
%    end
    if ~isempty(iDat),
      if ~isequal(varargout{iDat}.clab, varargcat{iDat}.clab),
        warning(['inconsistent clab structure will be repaired ' ...
                 'by using the intersection']); 
        commonclab= intersect(varargout{iDat}.clab, varargcat{iDat}.clab);
        varargout{iDat}= proc_selectChannels(varargout{iDat}, commonclab{:});
        varargcat{iDat}= proc_selectChannels(varargcat{iDat}, commonclab{:});
      end
      if ~isequal(varargout{iDat}.fs, varargcat{iDat}.fs)
        error('inconsistent sampling rate'); 
      end
      if ndims(varargout{iDat}.x)==2,    %% continuous data: concat signals
        varargout{iDat}.x= cat(1, varargout{iDat}.x, varargcat{iDat}.x);
      else                               %% data are epoched: append trials
        if size(varargout{iDat}.x,1)~=size(varargcat{iDat}.x,1), 
          error('inconsistent trial length');
        end
        varargout{iDat}.x= cat(3, varargout{iDat}.x, varargcat{iDat}.x);
      end
    end
    if ~isempty(iMrk),
      varargcat{iMrk}.pos= varargcat{iMrk}.pos + sum(T(1:ii-1));
      varargout{iMrk}= mrk_mergeMarkers(varargout{iMrk}, varargcat{iMrk});
    end
    if ~isempty(iMrkBV),
      if length(varargcat{iMrkBV})>1,
        orig_fs= eegfile_loadMatlab(file_list{ii-1}, 'vars','fs_orig');
        TT(ii-1)= round(T(ii-1)/varargcat{iNfo}.fs*orig_fs);
        shift= sum(TT(1:ii-1));
        for ii= 1:length(varargcat{iMrkBV}),
          varargcat{iMrkBV}(ii).pos= varargcat{iMrkBV}(ii).pos + shift;
        end
        varargout{iMrkBV}= cat(1, varargout{iMrkBV}, varargcat{iMrkBV});
      else
        varargcat{iMrkBV}.pos= varargcat{iMrkBV}.pos + sum(T(1:ii-1));
        varargout{iMrkBV}= ...
            mrk_mergeMarkers(varargout{iMrkBV}, varargcat{iMrkBV});
      end
    end
  end
end

if ~isempty(iDat),
  if ndims(varargout{iDat}.x)==2,
    varargout{iDat}.T= T;
  else
    varargout{iDat}.N= N;
  end
  if length(file_list)>1,
    varargout{iDat}.title= [varargout{iDat}.title ' et al.'];
    varargout{iDat}.file= strcat(opt.path, file_list);
  end
end

  
