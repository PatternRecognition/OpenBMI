function epo = proc_lapchannelwise(epo, fcn, varargin);
%PROC_LAPCHANNELWISE - APPLIES A PROC-FUNCTION CHANNELWISE TO LAP-CHANNELS
%
%Synopsis:
%  EPO = proc_lapchannelwise(EPO, PROC_FCN, ...)
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
%Note: 1) This function ignores further output arguments of the PROC-function.
%  2) The result of this function can be different from applying proc_laplace
%  first and then calling proc_channelwise, but these differences are tiny
%  and due to numerical particularities in matrix multiplication.
%
%See also: proc_channelwise

% adapted from Guido's proc_channelwise by Benjamin


%% find channel indices (idx) which can be laplace filtered
idx= [];
for ii = 1:length(epo.clab)
  clab= getClabForLaplace(epo, epo.clab{ii});
  if length(clab)>1,
    idx= [idx, ii];
  end
end

for k = 1:length(idx),
  ii = idx(k);
  clab= getClabForLaplace(epo, epo.clab{ii});
  
  ep = proc_selectChannels(epo, clab);
  ep = proc_laplace(ep);
  da = feval(['proc_' fcn], ep, varargin{:});
  
  if k==1,
    dat= copy_struct(da, 'not','x');
    sz= size(da.x);
    sz(2)= length(idx);
    dat.x= zeros(sz);
  end
  
  dat.x(:,k,:) = da.x(:,1,:);
  dat.clab{k} = da.clab{1};
end

epo = dat;
