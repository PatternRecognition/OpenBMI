function [toe, idx]= marker_mapping_SposRneg(desc)
%MARKER_MAPPING_SPOSRNEG - Map BV Markers to numeric values
%
%Synopsis:
%  [TOE, IDX]= marker_mapping_SposRneg(DESC)
%
%Arguments:
%  DESC - Cell of strings: BV markers, like {'S 23', 'R134'}
%
%Output:
%  TOE - Numeric representation of the type of event. S-markers are mapped
%      to positive and R-markers to negative values
%  IDX - Indices of mapped markers (non-S/R markers are discarded)

% 03-2011 Benjamin Blankertz


iS= strmatch('S', desc);
iR= strmatch('R', desc);
toe= zeros(size(desc));
toe(iS)= cellfun(@(x)(str2double(x(2:end))), desc(iS));
toe(iR)= -cellfun(@(x)(str2double(x(2:end))), desc(iR));
idx= find(toe);
toe= toe(idx);
