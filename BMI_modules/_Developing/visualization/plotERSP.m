function [ dat ] = plotERSP(data , varargin )
%UNTITLED 이 함수의 요약 설명 위치
%   자세한 설명 위치
%% It needs more modify code(?). please kindly wait until updating  
dat = data;
opt = opt_CellToStruct(varargin{:});
sampleFrequency=dat.fs;
%% mem model order?
% modelOrder = 18+round(sampleFrequency/100);
modelOrder = 20;
opt.lowPassCutoff=(sampleFrequency/2)-10;
% highPassCutoff=-1;
% freqBinWidth=2;
%  끝이 짤려서 줄임
opt.highPassCutoff=-1;
opt.trend=1;

    spectralSize = round(opt.spectralSize/1000 * sampleFrequency);
    spectralStep = round(opt.spectralStep/1000 * sampleFrequency);

    
% opt.spectralMovStep = fix((size(dat.x,1)-opt.spectralSize)/opt.spectralStep)+1;
% params = [modelOrder, opt.highPassCutoff+opt.freqBinWidth/2, opt.lowPassCutoff-opt.freqBinWidth/2, opt.freqBinWidth,round(opt.freqBinWidth/.2), ...
%           opt.trend, sampleFrequency, opt.spectralStep, (opt.spectralSize/opt.spectralStep)];
params = [modelOrder, opt.highPassCutoff+opt.freqBinWidth/2, opt.lowPassCutoff, opt.freqBinWidth,round(opt.freqBinWidth/.2), ...
          opt.trend, sampleFrequency];
% memparms = params;
% 
    params(8) = spectralStep;
    params(9) = spectralSize/spectralStep;

%% calculating

% C1_idx=find(dat.y_dec==1);
% C2_idx = find(dat.y_dec==2);
% C1=prep_selectTrials(dat,C1_idx);
% C2=prep_selectTrials(dat,C2_idx);
  C1.x = dat.x(:,1:8,:);
  


%% class 1
for trial=1:size(C1.x,2)
    plotData1=C1.x(:,trial,:);
    plotData1=squeeze(plotData1);
    [trialspectrum, freq_bins] = mem( plotData1, params );
    tmp=mean(trialspectrum, 3);
    C1Data(:,:,trial)=tmp;
end
freq_bins = freq_bins - opt.freqBinWidth/2;
C1plotData=mean(C1Data,3);
% dispmin=min(min(C1plotData));
% dispmax=max(max(C1plotData));

C1plotData=cat(2, C1plotData, zeros(size(C1plotData, 1), 1));
C1plotData=cat(1, C1plotData, zeros(1, size(C1plotData, 2)));

freq_bins(end+1) = freq_bins(end) + diff(freq_bins(end-1:end));

% Plot
figure(1);
% axis tight;
surf(freq_bins, [1:size(C1plotData, 2)], C1plotData(1:end,:)');
% surf(freq_bins, [1:size(C1plotData, 2)], C1plotData');
axis tight;
xlim([freq_bins(2) freq_bins(end-1)]); 
set(gca,'XTick',[freq_bins(2):2: freq_bins(end-1)]);

ylim([1 size(C1Data,2)]);
set(gca,'YTick',[1:1: size(C1Data,2)]);
% set(gca,'Yticklabel',char(dat.chan));
% set(gca,'YTick',[freq_bins(4):2: req_bins(end-1)]);
view(2);

colormap jet;
colorbar;
xlabel('Frequency [Hz]');
ylabel('Channels');
title('Frequency band-power per channel');



%% class 2 
% bci 2000 data
C2.x = dat.x(:,9:end,:);
for trial=1:size(C2.x,2)
    plotData2=C2.x(:,trial,:);
    plotData2=squeeze(plotData2);
    [trialspectrum, freq_bins] = mem( plotData2, params );
    tmp=mean(trialspectrum, 3);
    C2Data(:,:,trial)=tmp;
end
freq_bins = freq_bins - opt.freqBinWidth/2;
C2plotData=mean(C2Data,3);
% dispmin=min(min(C1plotData));
% dispmax=max(max(C1plotData));

C2plotData=cat(2, C2plotData, zeros(size(C2plotData, 1), 1));
C2plotData=cat(1, C2plotData, zeros(1, size(C2plotData, 2)));

freq_bins(end+1) = freq_bins(end) + diff(freq_bins(end-1:end));

% Plot
figure(2);
% axis tight;
surf(freq_bins(1:end-1), [1:size(C2plotData, 2)], C2plotData(1:end-1,:)');
% surf(freq_bins, [1:size(C1plotData, 2)], C1plotData');

xlim([freq_bins(1) freq_bins(end-1)]); 
set(gca,'XTick',[freq_bins(1):2: freq_bins(end-1)]);

ylim([1 size(C2Data,2)]);
set(gca,'YTick',[1:1: size(C2Data,2)]);
% set(gca,'Yticklabel',char(dat.chan));
% set(gca,'YTick',[freq_bins(4):2: req_bins(end-1)]);
view(2);

colormap jet;
colorbar;
xlabel('Frequency [Hz]');
ylabel('Channels');
title('Frequency band-power per channel');

%% r square value

  %% r square value
% ressq = calc_rsqu(double(avgdata1), double(avgdata2), 1);
% 
C1Data = double(C1Data); C2Data = double(C2Data);
for ch=1:size(C1Data, 2)
    for samp=1:size(C1Data, 1)
        ressq(samp, ch)=rsqu(C1Data(samp, ch, :), C2Data(samp, ch, :));
        amp1(samp, ch)=mean(C1Data(samp, ch, :));
        amp2(samp, ch)=mean(C2Data(samp, ch, :));
    end
end
ressq=cat(2, ressq, zeros(size(ressq, 1), 1));
ressq=cat(1, ressq, zeros(1, size(ressq, 2)));
ressq = ressq';
freq_bins(end+1) = freq_bins(end) + diff(freq_bins(end-1:end));

% Plot
figure(3);
% axis tight;
surf(freq_bins, [1:size(ressq, 2)], ressq(1:end,:)');
% surf(freq_bins, [1:size(C1plotData, 2)], C1plotData');

xlim([freq_bins(1) freq_bins(end-1)]); 
set(gca,'XTick',[freq_bins(1):2: freq_bins(end-1)]);

ylim([1 size(ressq,2)]);
set(gca,'YTick',[1:1: size(ressq,2)]);
% set(gca,'Yticklabel',char(dat.chan));
% set(gca,'YTick',[freq_bins(4):2: req_bins(end-1)]);
view(2);

colormap jet;
colorbar;
xlabel('Frequency [Hz]');
ylabel('Channels');
title('Frequency band-power per channel');

end

