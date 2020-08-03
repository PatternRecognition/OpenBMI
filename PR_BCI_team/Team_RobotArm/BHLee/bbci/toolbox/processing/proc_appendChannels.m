function dat= proc_appendChannels(dat, dat_append,varargin)
%dat= proc_appendChannels(dat, dat_append,<chanName>)
%
% IN   dat        - data structure of continuous or epoched data
%      dat_append - same data structure as dat, to be appended.
%                   Alternatively a single vector or an N x p matrix, where
%                   N is the length of the dat.x field and p is the number
%                   of new channels. For epoched data, N x ep or N x p x ep
%                   where ep is the number of epochs.
%      chanName   - If dat_append is a vector, chanName is the name to be
%                   entered in the .clab field. If multiple channels are
%                   added, chanName should be a cell array of strings.
%
% OUT  dat        - a miracle

% bb, ida.first.fhg.de

if isempty(dat)
    dat = dat_append;
    return
elseif isempty(dat_append)
    return
elseif isstruct(dat_append)
%% dat_append is a struct
  if isfield(dat,'t') && iscell(dat.t)
      data_dim = size(dat.t,2);
  else
      data_dim = 1;
  end;

  dat.x= cat(data_dim+1, dat.x, dat_append.x);
  dat.clab= cat(2, dat.clab, dat_append.clab);

  if isfield(dat,'p')
    if data_dim ~= 1
      warning('not yet tested with data_dim ~= 1 !!! remove this warning if it works, fix it otherwise.')
    end
    dat.p = cat(data_dim+1, dat.p, dat_append.p);
  end
  if isfield(dat,'V')
    if data_dim ~= 1
      warning('not yet tested with data_dim ~= 1 !!! remove this warning if it works, fix it otherwise.')
    end
    dat.V = cat(data_dim+1, dat.V, dat_append.V);
  end

else
%% dat_append is a vector or matrix
  if nargin<=2 , error('Provide names for new channels'),end
  chanName = varargin{1};
  if ~iscell(chanName); chanName = {chanName}; end
  dat.x = cat(2,dat.x,dat_append);
  dat.clab = {dat.clab{:} chanName{:}};
end

