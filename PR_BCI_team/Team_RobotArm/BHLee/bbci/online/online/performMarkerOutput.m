function mrkOut = performMarkerOutput(modus,varargin);
%PERFORMMARKEROUTPUT matches marker to outputs
% 
% usage:
% ('init')      mrkOutDef = performMarkerOutput('init',opt,mrkOutDef);
% ('apply')     mrkOut = performMarkerOutput('apply',mrkOutDef,packetLength,timestamp);
%
% input:
%   opt         opt is not needed so far
%   mrkOutDef   a struct array with fields
%               .marker    marker to use 
%               .value     values to map (numeric array regarding 
%                          length of the marker cell array
%               .no_marker default value if no marker exists
%   packetLength   the length of the actual package
%   timestamp      the actual timestamp
%
% output:
%   mrkOut      a cell array with 3dim entries for each mrkOutDef. 
%               1. the value
%               2. the timestamp of the marker
%   mrkOutDef   mrkOutDef with all defaults
%
% description:
% init: initialize this program
% apply: get a result
%
% see bbci_bet_apply and adminMarker
%
% Guido Dornhege
% TODO: extended documentation by Schwaighase
% $Id: performMarkerOutput.m,v 1.2 2006/05/02 13:43:17 neuro_cvs Exp $

switch modus
 case 'init'
  %INIT
  opt = varargin{1};
  mrkOut = varargin{2};
  for i = 1:length(mrkOut)
    mrkOut(i) = set_defaults(mrkOut(i),'no_marker',0,...
                                       'value',1);
    if ~iscell(mrkOut(i).marker)
      mrkOut(i).marker = {mrkOut(i).marker};
    end
  end
 
 case 'apply'
  %APPLY
  mrkOutDef = varargin{1};
  if isempty(mrkOutDef) || isempty([mrkOutDef.value])
    mrkOut = {};
  else
    packetLength = varargin{2};
    timestamp = varargin{3};
    mrkOut = cell(1,length(mrkOutDef));
    [toe,time] = adminMarker('query',[-packetLength 0]);
    for i = 1:length(mrkOutDef)
      for j = 1:length(mrkOutDef(i).marker);
          ii = mrkMatch(mrkOutDef(i).marker{j},num2cell(toe));
          for k = 1:length(ii)
            mrkOut{i} = [mrkOut{i};mrkOutDef(i).value(j);time(ii(k))+timestamp];
          end
      
      end
      
      if isempty(mrkOut{i})
        mrkOut{i} = [mrkOutDef(i).no_marker;timestamp];
      end
    end
  end
 otherwise 
  error('Unknown case');
end

return;
