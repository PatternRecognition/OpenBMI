function mrk = reduceMarkerToken(mrk,bit);
%REDUCEMARKERTOKEN REDUCES ALL TOKEN AT THE SPECIFIED BITS
%
% usage:
%  mrk = reduceMarkerToken(mrk,bit);
%
% input:
%  mrk      a marker structure 
%  bit      the bits to reduce

% output:
%  mrk       a marker structure with others tokens
%
% Guido DOrnhege, 02/09/2004

to = dec2bin(abs(mrk.toe),8);
to(:,bit) = '0';
mrk.toe = sign(mrk.toe).*transpose(bin2dec(to));





