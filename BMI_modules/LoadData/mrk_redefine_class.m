function [ mrk ] = mrk_redefine_class(mrk, mrk_define )
%MRK_REDIFINE_CLASS Summary of this function goes here
%   Detailed explanation goes here
mrk_define=opt_proplistToCell(mrk_define{:});
% delete undefined classes
nc_all=logical(1:length(mrk.y));
for i=1:length(mrk_define)
    [nc]=find(mrk.y==str2num(mrk_define{i}));
    for j=1:length(nc)
        mrk.class{nc(j)}=mrk_define{i,2};
    end
    nc_all(nc)=0;
end
mrk.y=mrk.y(~nc_all);
mrk.t=mrk.t(~nc_all);
mrk.class=mrk.class(~nc_all);
nClasses=length(mrk_define);
mrk.logical_y= zeros(nClasses, numel(mrk.y));
mrk.dClasses=cell(size(mrk_define));
mrk.nClasses=nClasses;

%logical Y lable
for i=1:length(mrk_define)
    c_n=str2num(cell2mat(mrk_define(i)));
    [temp idx]=find(mrk.y==c_n);
    mrk.logical_y(i,idx)=1;    
    mrk.dClasses(i,:)=mrk_define(i,:);
end

end

