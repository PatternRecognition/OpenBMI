% PROC_CSD - calculates current-source density estimation
%
% Description:
% proc_csd calculates for all epochs of each class 
% the current-source density estimation
%
% Usage:
%   [DAT] = proc_pr_trPCA(DAT, <OPT>);
%
% Input:
%   DAT: data structure of epoched data
%   
% OPT: struct or property/value list of optional properties:
%   'mntfile':  path to file containing spherical coordinate system
%   'appendix': if 1, appends string '_csd' to channel labels (default 1)
%
% Output:
%   DAT: csd transformed data
%
% CSD() is part of the CSDToolbox
% Kayser, J. (2009). Current source density (CSD) interpolation 
% using spherical splines - CSD Toolbox (Version 1.0) 
% [http://psychophysiology.cpmc.columbia.edu/Software/CSDtoolbox]. 
% New York State Psychiatric Institute: Division of Cognitive Neuroscience.
%
% Example:
%   epo_csd= proc_csd(epo)


function [csddat] = proc_csd(dat, varargin)

global BCI_DIR
global last_clab G H

bbci_warning('in construction', 'construction-site')

default_mntfile_folder=  [BCI_DIR 'import/CSDtoolbox/resource/'];
%default_mntfile_folder=  [BCI_DIR 'toolbox/data/config/'];

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'showmnt', 0, ...
                  'mntfile', [default_mntfile_folder ...
                              '10-5-System_Mastoids_EGI129.csd'], ...
                   'appendix',1);
%                 'mntfile', 'bbci');

if ~exist('GetGH','file'),
  path(path, [BCI_DIR 'import/CSDtoolbox/func']);
end

if isempty(last_clab) | ~isequal(dat.clab, last_clab),
  [clab]= textread(opt.mntfile, '%s%*[^\n]', 'commentstyle','c++');
  notfound= find(~ismember(dat.clab, clab));
  if ~isempty(notfound),
    warning('no positions found for the following channels: %s.', vec2str(dat.clab(notfound)));
    dat= proc_selectChannels(dat, 'not',dat.clab(notfound));
  end
  last_clab = dat.clab;
  
  M= ExtractMontage(opt.mntfile, dat.clab');
  if opt.showmnt,
    MapMontage(M);
  end
  [G,H]= GetGH(M);
end

if isfield(dat, 'y') % epoched data
  nClasses= size(dat.y,1);    
  for n=1:nClasses
    classInd{n,:} = find(dat.y(n,:));
  end
end
T = size(dat.x,1);

csddat = dat;
%% BB: This looks strange. I guess the loop can be replaced.
%% Furthermore, the function should also be applicable to cnt data.
if isfield(dat, 'y') % epoched data
  for n = 1:nClasses,
    for t = 1:T,
      D = reshape(dat.x(t,:,classInd{n}), ...
                  [size(dat.x,2) size(dat.x(t,:,classInd{n}),3)]);
      csddat.x(t,:,classInd{n}) = CSD(D, G, H);
    end
  end
else % cnt data
  csddat.x = CSD(dat.x', G, H)';
end

if opt.appendix
  csddat.clab= cprintf('%s csd', dat.clab);
end
