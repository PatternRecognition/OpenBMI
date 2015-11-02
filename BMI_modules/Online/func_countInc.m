function [ out ] = func_countInc( num )
%FUNC_COUNTINC Summary of this function goes here
%   Detailed explanation goes here
if isstr(num)
    tNum=str2num(num)+1;
    out=num2str(tNum);
end
if isnumeric(num)
    tNum=num+1;
    out=tNum;
end
end

