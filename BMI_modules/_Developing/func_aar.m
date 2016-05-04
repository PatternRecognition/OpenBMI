function dat=func_aar(dat,varargin)
%Calculating the adaptive auto regression parameters(AAR) and adaptive auto
%moving average auto regrssion parameters(AARMA)
%
% The AAR process is described as following
%       y(k) - a(k,1)*y(t-1) -...- a(k,p)*y(t-p) = e(k);
% The AARMA process is described as following
%       y(k) - a(k,1)*y(t-1) -...- a(k,p)*y(t-p) = e(k) + b(k,1)*e(t-1) + ... + b(k,q)*e(t-q);

%In dat      -     input the data structure of OpenBMI segementation original data
%   varadgin -     model selection for obtatining AAR parameter
%                  Mode : [amode vmode] (default [1,2]) (see also aar function)
%                         amode : upadating co-varaince matrix method (ranges 1 to 12)
%                         vmode : estimating the innovation variance method (ranges 1 to 7)
%                  order: model order [p,(q)] of AAR or AARMA  (default [10,0])
%                         AAR(p) order as [p] and AARMA(p,q) as [p,q]
%                  UC   : Update coefficeint (default 0.0085)
%out dat     -     data structure of otanined AAR or AARMA parameter in OpenBMI sturcture

% Example code Feature=func_aar(EMT,{'Mode',[11 2],'order', [5 2],'UC', 0.003)};
% Example code
if isempty(dat)
    warning('[OpenBMI] Warning! data is empty.');
end

opt=opt_cellToStruct(varargin{:});
epo=struct('Mode',[],'order',[],'UC',[]);

% opt=struct('Mode',[],'order',[]);
if ~isfield(opt,'Mode')
    warning('[OpenBMI] Warning! Mode is empty. default model is v1,a1');
    epo.Mode=[1,2];
else
    epo.Mode=opt.Mode;
end
if ~isfield(opt,'order')
    warning('[OpenBMI] Order is not exist. default order is [2,0]');
    epo.order=[2,0];
else
    epo.order=opt.order;
end
order=epo.order(1);

if ~isfield(opt,'UC')
    warning('[OpenBMI] Update coefficient is not exist. default order is 0.0085');
    epo.UC=0.0085;
else
    epot.AR=opt.AR;
end

[T, nEvents , nChans]= size(dat.x);
tp_ar= [];
Dat=reshape(dat.x, [T, nEvents*nChans]);

%% optain AAR, AARMA parameter
for i= 1:nChans*nEvents,
    aar1=aar(Dat(:,i),epo.Mode,epo.order,epo.UC);
    tp_aar(:,i)=aar1(end,:);
end
%% reshaping the parameter as order*channel by trials
Dat=permute(reshape(tp_aar, [order, nEvents,nChans]), [1 3 2])
Dat=reshape(Dat, [order* nChans nEvents])

dat.x= Dat;

