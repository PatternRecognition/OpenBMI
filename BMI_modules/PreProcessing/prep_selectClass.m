function [ out ] = prep_selectClass( dat, varargin )
%MRK_SELECT_CLASS Summary of this function goes here
%   Detailed explanation goes here

opt=opt_cellToStruct(varargin{:});

if nargin==0
    error('parameter is missing');
end

if ~isfield(opt, 'class')
    error('parameter is missing: marker.class');
end

if ndims(dat.x)==2
    type='cnt';
elseif ndims(dat.x)==2
    type='epo';
end

[n_c nn]=size(dat.class);
n_c=zeros(1, n_c);

%find class index
if ischar(opt.class) % one class
    temp=ismember(dat.class(:,2),opt.class);
    [a b]=find(temp==1);
    n_c(a)=1;
    clear a b;
else
    for i=1:length(opt.class)
        temp=ismember(dat.class(:,2),opt.class{i});
        [a b]=find(temp==1);
        n_c(a)=1;
        clear a b;
    end
end

[n_d]=find(n_c==0);
del_classes=cell(length(n_d),1);

for i=1:length(n_d)
    del_classes{i}=dat.class{n_d(i),2};
end


out=prep_removeClass(dat,del_classes);


end