function dat_out=func_featureSelection_filterbank(dat_in,varargin)

if iscell(varargin)
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) 
    opt=varargin{:};
end
idx= opt.index;

out=dat_in{1};
out.x=[];

[j,k]=size(idx);
for ii=1:j
    out.x(ii,:)=dat_in{idx(ii,1)}.x(idx(ii,2),:);
end

dat_out=out;

end