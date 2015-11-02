function [ mrk ] = mrk_remove_class(  mrk_orig, varargin )
%MRK_REMOVE_CLASS Summary of this function goes here
%   Detailed explanation goes here

if ~iscellstr(varargin)
    warning('check the class parameter, it shold be a string');
end

if iscellstr(varargin)
    n_c=~ismember(mrk_orig.class,varargin);
    mrk.y=mrk_orig.y(n_c);
    mrk.t=mrk_orig.t(n_c);
    mrk.class={mrk_orig.class{n_c}};
end

% Delete the classes in marker.dClasses
c_n=zeros(1,mrk_orig.nClasses);
for i=1:mrk_orig.nClasses
   if strcmp(lower(mrk_orig.dClasses{i,2}),lower(varargin))
   c_n(i)=1;
   end
end
[a b]=find(c_n==0);
mrk.dClasses=mrk_orig.dClasses(b,:);
mrk.nClasses=length(mrk.dClasses(:,1));

if isfield(mrk_orig, 'logical_y')
    %logical Y lable
    for i=1:mrk.nClasses
        c_n=str2num(cell2mat(mrk.dClasses(i)));
        [temp idx]=find(mrk.y==c_n);
        mrk.logical_y(i,idx)=1;
    end
end

end

