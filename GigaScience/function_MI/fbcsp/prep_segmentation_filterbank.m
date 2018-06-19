function [ dat_out ] = prep_segmentation_filterbank( dat_in, varargin )

if iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) 
    opt=varargin{:};
end

time_interval=opt.interval;
num_cnt=length(dat_in);

for ii=1:num_cnt
    SMT{ii} = prep_segmentation(dat_in{ii}, {'interval', time_interval});
end

dat_out=SMT;

end