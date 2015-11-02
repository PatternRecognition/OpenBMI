function [ mrk ] = mrk_define_class( mrk_orig ,mrk_define )
%MRK_DEFINE_CLASS Summary of this function goes here
%   Detailed explanation goes here

for i=1:length(mrk_define)
    [nc]=find(mrk_orig.y==str2num(mrk_define{i}));
    for j=1:length(nc)
        mrk_orig.class{nc(j)}=mrk_define{i,2};
    end
end

% delete undefined classes
n_c=~ismember(mrk_orig.class,'Stimulus');
mrk.y=mrk_orig.y(n_c);
mrk.t=mrk_orig.t(n_c);
mrk.class={mrk_orig.class{n_c}};

nClasses=length(mrk_define);
mrk.logical_y= zeros(nClasses, numel(mrk.y));
mrk.dClasses=cell(size(mrk_define));

%logical Y lable
for i=1:length(mrk_define)
    c_n=str2num(cell2mat(mrk_define(i)));
    [temp idx]=find(mrk.y==c_n);
    mrk.logical_y(i,idx)=1;    
    mrk.dClasses(i,:)=mrk_define(i,:)
end

end

