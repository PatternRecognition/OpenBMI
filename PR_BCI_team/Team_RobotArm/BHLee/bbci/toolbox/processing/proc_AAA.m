function [cnt, B,C] = proc_AAA(cnt,cnt_cal, mrk_cal, channel,window, basewindow, opt)
% [cnt,B] = proc_MSEC(cnt,cnt_cal,mrk_cal,<channel = 'EOGv',window = [-500 500], basewindows = 0.1*length(window), opt.mean = 1>)
%
% INPUT:
% cnt is the usual cnt, or epo struct
% cnt_cal is the calibration data set
% mrk_cal are the markers of relevant activation (array)
% channel is the relevant channel of the EOG
% window is an array which describes the window around the peak
% basewindow is the length of the windows, where means are calculated and subtracted
% opt.mean is a flag, if it is true the mean of all features is determined before calculating SourceVectors, else on all intervals SourceVectors are determined.
%
% OUTPUT:
% cnt = the corrected Waveform
% B = the correlation coefficient
% C = the correction ocefficient
%
% Guido Dornhege
% 14.08.2002

% setting defaults

if nargin<3
     error('not enough input arguments');
end

if ~exist('channel') | isempty(channel)
     channel = 'EOGv';
end

if ~exist('window') | isempty(window)
     window = [-500 500];
end

if ~exist('basewindow') | isempty(basewindow)
     basewindow = 0.1*(window(2)-window(1));
end

if ~exist('opt') | isempty(opt)
     opt.mean = 1;
end

if ~isfield(opt,'mean')
     opt.mean = 1;
end


%calculating B
index = find(strcmp(cnt_cal.clab,channel));

if isempty(index)
     error('the channel does not exist');
end

window = round(cnt.fs*window/1000);
nu = window(2)-window(1);
nu = nu/basewindow;
nuWin = round(nu);
nu = nuWin*basewindow;
window = floor(0.5*(window(2)+window(1))+0.5*[-nu,nu]);

cali = [];
for i = mrk_cal
   cali = cat(3,cali,cnt_cal.x((window(1):window(2)-1)+i,:));
end

if opt.mean
     cali = mean(cali,3);
else
     calim = mean(cali,3);
     cali = permute(cali,[2 1 3]);
     cali = reshape(cali,[size(cali,1), size(cali,2)*size(cali,3)]);
     cali = cali';
end




mecal = permute(cali,[2 1]);
mecal = reshape(cali,[size(mecal,1),size(mecal,2)/nuWin,nuWin]);
mecal = mean(mecal,3);
mecal = repmat(mecal,[1 nuWin]);
mecal = permute(mecal,[2 1]);

cali = cali-mecal;


B = cali'*cali(:,index);
B = B/(B(index));

C = mean(mecal(:,index)-mecal*B);

for i = 1:size(cnt.x,3)
     cnt.x(:,:,i) = cnt.x(:,:,i)-cnt.x(:,index,i)*B'-C;
end
