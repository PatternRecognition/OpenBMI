function [ dat_out, idx ] = func_MIBIF_filterbank( dat_in, varargin )

if iscell(varargin)
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) 
    opt=varargin{:};
end

num_feat=opt.nFeatures;

for ii=1:length(dat_in)
    [dat_out{ii}]=func_MIBIF(dat_in{ii},{'nFeatures',num_feat});
end

mutual_info=[];
for ii=1:length(dat_out)
    mutual_info=[mutual_info dat_out{ii}.mutual_info];
end

[j,k]=size(mutual_info);
mutual_info_vect=reshape(mutual_info,[j*k,1]);
[mi.value,mi.indx]=sort(mutual_info_vect,'descend');

for ii=1:num_feat
    xx=floor((mi.indx(ii)-1)/j)+1;
    yy=mod((mi.indx(ii)),j);
    if yy==0
        yy=j;
    end
    idx_selected{ii}=[xx yy];
    idx_pair{ii}=[xx j-yy+1];
end
idx = union(cell2mat(idx_selected(:)),cell2mat(idx_pair(:)),'rows');

out=dat_out{1};
out.mutual_info=mi.value(1:num_feat);
out.idx=idx;
out.x=[];

[j,k]=size(idx);
for ii=1:j
    out.x(ii,:)=dat_out{idx(ii,1)}.x(idx(ii,2),:);
end
dat_out=out;
end