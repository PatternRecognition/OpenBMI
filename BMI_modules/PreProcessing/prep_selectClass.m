function [ marker ] = prep_selectClass( marker, classes )
%MRK_SELECT_CLASS Summary of this function goes here
%   Detailed explanation goes here

if nargin==0
    error('parameter is missing');
end

if ~isfield(marker, 'class')
    error('parameter is missing: marker.class');
end

n_c=zeros(1,marker.nClasses);

%find class index
for i=1:length(classes)
    temp=ismember(marker.class,classes{i});
    [a b]=find(temp(:,2)==1);
    n_c(a)=1;
    clear a b;
end

[n_d]=find(n_c==0);
del_classes=cell(length(n_d));

for i=1:length(n_d)
    del_classes{i}=marker.class{n_d(i),2};
end

for i=1:length(n_d)
    marker=prep_removeClass(marker,del_classes{i});
end

end