function [ mrk ] = mrk_select_class( mrk, classes )
%MRK_SELECT_CLASS Summary of this function goes here
%   Detailed explanation goes here
n_c=zeros(1,mrk.nClasses);

for i=1:length(classes)
    temp=ismember(mrk.dClasses,classes{i});
    [a b]=find(temp(:,2)==1);
    n_c(a)=1;
    clear a b;
end

[n_d]=find(n_c==0);
del_classes=cell(length(n_d));

for i=1:length(n_d)
    del_classes{i}=mrk.dClasses{n_d(i),2};
end

for i=1:length(n_d)
    mrk=mrk_remove_class(mrk,del_classes{i});
end

end
