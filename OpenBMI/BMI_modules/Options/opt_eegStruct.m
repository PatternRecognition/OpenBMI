function [ cnt ] = opt_eegStruct( dat, field )
%OPT_EEGSTRUCT Summary of this function goes here
%   Detailed explanation goes here

%input: cell structure, {struct1, struct2, struct3}
% Select necessary fields from input structures 
% construct initial OpenBMI data structure

for i=1:length(field)
    [cnt(:).(field{i})]=[];
end

stc=struct;
if iscell(dat)
    nC=length(dat);
    for i=1:nC
        t_stc=opt_selectField(dat{i},field);
        stc=opt_catStruct(stc,t_stc);
    end
end

for i=1:length(field)
    if ~isempty(stc.(field{i}))
        cnt(:).(field{i})= stc.(field{i});     
    else
        warn_str=sprintf('OpenBMI: The field "%s" is empty', field{i});
        disp(warn_str);
    end
end

end

