function [ out ] = opt_selectField( in, field )
%OPT_DELECTFEIDL Summary of this function goes here
%   Detailed explanation goes here
% in: structure
% select necessary field from input structure "in"

for i=1:length(field)
    if isfield(in,field{i})
        out.(field{i})=in.(field{i});
    end    
end

end

