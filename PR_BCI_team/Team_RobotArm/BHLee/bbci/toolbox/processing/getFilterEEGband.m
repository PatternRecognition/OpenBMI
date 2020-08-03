function [b,a]= getFilterEEGband(band, fs)
%[b,a]= getFilterEEGband(bandName, fs)
%
% bandname in {'infraDelta',  %   <=2.5
%              'SCP',         % 0.2-2.5
%              'theta',       %   3-6.5
%              'alpha',       %   9-12
%              'alpha1',      %   7-10
%              'alpha2',      %  10-13
%              'beta',        %  15-30
%              'beta1',       %  16-24
%              'beta2',       %  24-30
%              'alphaBeta',   %   7-29
%              'alpha2Beta2', %   7-30
%              'gamma',       %  40-45
%              'cut50',       %    <45
%              'band50',      % 0.2-45
%              'emg',         %  20-200
%              'raw',''};

band(find(ismember(band, '_\')))= [];
switch lower(band),
 case {'raw', ''},
  b= []; a= [];
 case 'scp',
  Bp= [0.2 2.5]; Bs=[0.1 6]; Rp=0.5; Rs=20;
 case 'infradelta',
  [b,a]= ellip(4, 1, 60, 2.5*2/fs);
 case 'theta',
  Bp= [3 6.5]; Bs= [2 7.5]; Rp=0.5; Rs= 35;
 case 'alpha',
  Bp= [9 12]; Bs= [8 14]; Rp=0.1; Rs= 35;
 case 'alpha1',
  Bp= [7 10]; Bs=[6 11]; Rp=1; Rs= 35;
 case 'alpha2',
  Bp= [10 13]; Bs=[9 14]; Rp=1; Rs= 35;
 case 'beta',
  Bp= [15 30]; Bs= [14 32]; Rp=0.5; Rs= 35;
 case 'beta1',
  Bp= [16 22]; Bs= [15 24]; Rp=0.1; Rs= 35;
 case 'beta2',
  Bp= [24 30]; Bs= [22 32]; Rp=0.1; Rs= 35;
 case 'alphabeta',
  Bp= [7 29]; Bs= [1 32]; Rp=1; Rs=35;
 case 'alpha2beta2',
  Bp= [10 30]; Bs= [8 34]; Rp=1; Rs=35;
 case 'gamma',
  Bp= [40 45]; Bs= [39 47]; Rp=1; Rs= 35;
 case 'cut50',
  [b,a]= ellip(10, 0.1, 80, 45*2/fs);
 case 'band50',
  Bp= [0.24 46.4]; Bs=[0.1 55]; Rp=2; Rs=20;
 case 'emg',
  if fs>=600,
    Bp= [20 200]; Bs= [15 250]; Rp=0.1; Rs=48;
  elseif fs>=500,
    Bp= [20 200]; Bs= [15 240]; Rp=0.1; Rs=48;
  elseif fs>=100,
    Bp= [20 45]; Bs= [15 48]; Rp=0.1; Rs=48;
  else
    error('emg filter requires a sampling rate of at least 100 Hz');
  end    
 otherwise,
  error('unknown band');
end

if ~exist('b', 'var'),
  [n, Wn]= ellipord(Bp*2/fs, Bs*2/fs, Rp, Rs);
  [b, a]= ellip(n, Rp, Rs, Wn);
end
