function [S,HDR] = ukbfread(HDR);
% UKBFREAD loads the signal file
%
% [S,HDR] = ukbfread(HDR);
%
% see ukbfopen

% This code is extracted from sread.m of BIOSIG-toolbox http://biosig.sf.net/

maxsamples = HDR.AS.endpos-HDR.FILE.POS;
S = []; count = 0;
while maxsamples>0,	
  [s,c]  = fread(HDR.FILE.FID,[HDR.NS,min(2^16,maxsamples)], HDR.GDFTYP);
  count = count + c;
  maxsamples = maxsamples - c/HDR.NS;
  if c,
    S = [S;s(HDR.InChanSelect,:)'];
  end;
end;
HDR.FILE.POS = HDR.FILE.POS + count/HDR.NS;

S = double(S) * HDR.Calib(2:end,:);
for k = 1:size(HDR.Calib,2),
  S(:,k) = S(:,k) + HDR.Calib(1,k);
end;

