function C= train_parzen(x,y,sigma)
%TRAIN_PARZEN trains a Parzen window approach for classication.
%
% usage:
%   C= train_parzen(x,y,sigma)
%
% inputs:
%   x       training data
%   y       training labels
%   sigma   kernel width, thus the kernel is 
%                  k(xi,xj) = exp(-(xi-xj)'*(xi-xj)/sigma)
%
% outputs:
%   C       the trained classifier
%
% sth * 25nov2004

C.x = x;
C.y = y;
C.sigma = sigma;
