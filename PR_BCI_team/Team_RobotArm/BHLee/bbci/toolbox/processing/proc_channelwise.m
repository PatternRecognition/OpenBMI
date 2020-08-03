function dat = proc_channelwise(epo, fcn, varargin);
%PROC_CHANNELWISE - APPLIES A PROC-FUNCTION CHANNELWISE
%
%Synopsis:
%  EPO = proc_channelwise(EPO, PROC_FCN, ...)
%
%Arguments:  
%  EPO:   Data structure, channels in second dimension of epo.x
%         May be continuous or epoched data.
%  PROC_FCN: String. Name of function to call ('proc_' is prepended)
%  ...    All further arguments are passed directly to the PROC-function.
%
%Returns:
%  EPO:   Struct of processed data.
% 
%Note: This function ignores further output arguments of the PROC-function.
%
%See also: proc_lapchannelwise

% Guido Dornhege, 23/02/05
%  + modifications by Benjamin
% $Id: proc_channelwise.m,v 1.4 2007/09/24 10:16:34 neuro_cvs Exp $

chan= 1:size(epo.x,2);
if iscell(fcn),
  fcn= chanind(epo, fcn);
end
if isnumeric(fcn),
  chan= fcn;
  fcn= varargin{1};
  varargin= varargin(2:end);
end
ep = proc_selectChannels(epo, chan(1));
dat = feval(['proc_' fcn], ep, varargin{:});
N = size(dat.x,2)/size(ep.x,2);

if isequal(size(dat.x), size(ep.x))
  overwr = 1;
  epo.x(:,chan(1),:) = dat.x(:,1,:);
  if isfield(epo, 'p')
    epo.p(:,chan(1),:) = dat.p(:,1,:);
  end
  if isfield(epo, 'V')
    epo.V(:,chan(1),:) = dat.V(:,1,:);
  end
else
  overwr = 0;
  sz= size(dat.x);
  sz(2)= length(epo.clab)*N;
  x = dat.x;
  dat.x=  zeros(sz);
  dat.x(:,1:N,:) = x;
  clear x;
  if isfield(dat, 'p')
    p = dat.p;
    dat.p=  zeros(sz);
    dat.p(:,1:N,:) = p;
    clear p;
  end
  if isfield(dat, 'V')
    V = dat.V;
    dat.V=  zeros(sz);
    dat.V(:,1:N,:) = V;
    clear V;
  end    
end

for ii = 2:length(chan)
  ep = proc_selectChannels(epo, chan(ii));
  da = feval(['proc_' fcn], ep, varargin{:});
  if overwr
    epo.x(:,chan(ii),:) = da.x(:,1,:);
    if isfield(epo, 'p')
      epo.p(:,chan(ii),:) = da.p(:,1,:);
    end
    if isfield(epo, 'V')
      epo.V(:,chan(ii),:) = da.V(:,1,:);
    end
    epo.clab{chan(ii)} = da.clab{1};
  else
    idx = (chan(ii)-1)*N+1:chan(ii)*N;
    dat.x(:,idx,:) = da.x;
    if isfield(dat, 'p')
      dat.x(:,idx,:) = da.x;
    end
    if isfield(dat, 'V')
      dat.V(:,idx,:) = da.V;
    end
    dat.clab(idx) = da.clab;
  end
end

if overwr,
  dat.x= epo.x;
  if isfield(epo, 'p')
    dat.p = epo.p;
  end
  if isfield(epo, 'V')
    dat.V = epo.V;
  end
  dat.clab= epo.clab;
end
%epo = dat;
