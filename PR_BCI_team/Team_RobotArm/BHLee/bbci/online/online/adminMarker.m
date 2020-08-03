function [toe,timeshift] = adminMarker(modus,varargin);
%ADMINMARKERS ADMINISTRATES ALL MARKERS FOR BBCI_BET_APPLY
%
% usage:
% ('init'):       adminMarker('init',opt);
% ('add'):        adminMarker('add',time,pos,toe);
% ('query'):  [toe,timeshift] = adminMarker('query',ival);
%
% input:
%      time    the actual timestamp relatively to start
%      pos     positions of markers regarding time (should be negative)
%      toe     the tokens of markers regarding time 
%      ival    the ival relatively to timestamp the marker should be.
%      
% output:
%      toe         the marker tokens
%      timeshift   the timeshifts of the markers
%
% see bbci_bet_apply and getCondition
%
% Guido Dornhege, 02/12/2004
% TODO: extended documentation by Schwaighase 
% $Id: adminMarker.m,v 1.1 2006/04/27 14:22:08 neuro_cvs Exp $

persistent mrk_queue timestamp opt

switch modus
 case 'init'
  % INITIALIZATION
  opt= propertylist2struct(varargin{:});
  opt= set_defaults(opt,'log',1,...
                        'mrkQueueLength',100);
  mrk_queue = cell(3,opt.mrkQueueLength);
  for i = 1:opt.mrkQueueLength
    mrk_queue{1,i} = nan;
    mrk_queue{2,i} = nan;
  end
  timestamp = 0;
  
 case 'add'
  % ADD MARKERS
  time = varargin{1};
  pos = varargin{2};
  toe = varargin{3};
  desc = varargin{4};
  timestamp = time;
  for i = 1:length(pos)
      if strcmp(desc{i},'Stimulus') | strcmp(desc{i},'Response')
          symb = str2double(toe{i}(2:end));
      else
          symb = toe{i};
      end
    mrk_queue = cat(2,mrk_queue(:,2:end),{timestamp+pos(i); symb; desc{i}});
    if opt.log 
      writeClassifierLog('marker', timestamp+pos(i), symb, desc{i});
    end
  end
  
 case 'query'
  % QUERY MARKERS
  ival = varargin{1};
  z = [mrk_queue{1,:}] - timestamp;
  ind = find(z>ival(1) & z<=ival(2));
  toe = [mrk_queue{2,ind}];
  timeshift = z(ind);
  
 otherwise
  error('Unknown case');
end
