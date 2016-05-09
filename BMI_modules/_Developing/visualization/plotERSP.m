function [ dat ] = plotERSP(data , varargin )
%UNTITLED 이 함수의 요약 설명 위치
%   자세한 설명 위치
%% It needs more modify code(?). please kindly wait until updating
%% Load the data
dat = data;
opt = opt_cellToStruct(varargin{:});
sampleFrequency=dat.fs;
%% Setting the parameter
% modelOrder = 18+round(sampleFrequency/100);
modelOrder = 20;
opt.lowPassCutoff=(sampleFrequency/2)-10;
opt.freqBinWidth= 2;opt.spectralSize= 333;opt.spectralStep = 166;
opt.highPassCutoff=-1;
opt.trend=1;
opt.spectralSize = round(opt.spectralSize/1000 * sampleFrequency);
opt.spectralStep = round(opt.spectralStep/1000 * sampleFrequency);
% opt.spectralMovStep = fix((size(dat.x,1)-opt.spectralSize)/opt.spectralStep)+1;
params = [modelOrder, opt.highPassCutoff+opt.freqBinWidth/2, opt.lowPassCutoff, opt.freqBinWidth,round(opt.freqBinWidth/.2), ...
    opt.trend, sampleFrequency];
params(8) = opt.spectralStep;
params(9) = opt.spectralSize/opt.spectralStep;

%% Separating the data per classes
C1_idx=find(dat.y_dec==1);
C2_idx = find(dat.y_dec==2);
C1=prep_selectTrials(dat,C1_idx);
C2=prep_selectTrials(dat,C2_idx);
%%
if strcmp(opt.Xaxis, 'Frequency') == 1 && strcmp(opt.Yaxis, 'Channel') == 1
    %% class 1
    for trial=1:size(C1.x,2)
        plotData1=C1.x(:,trial,:);
        plotData1=squeeze(plotData1);
        [trialspectrum, C1_freqBins] = mem( plotData1, params );
        tmp=mean(trialspectrum, 3);
        C1Data(:,:,trial)=tmp;
    end
    C1_freqBins = C1_freqBins - opt.freqBinWidth/2;
    C1plotData=mean(C1Data,3);
    
    C1plotData=cat(2, C1plotData, zeros(size(C1plotData, 1), 1));
    C1plotData=cat(1, C1plotData, zeros(1, size(C1plotData, 2)));
    
    C1_freqBins(end+1) = C1_freqBins(end) + diff(C1_freqBins(end-1:end));
    
    % Plot
    figure(1);
    surf(C1_freqBins, [1:size(C1plotData, 2)], C1plotData(1:end,:)');
    axis tight;
    
    set(gcf,'Renderer','Zbuffer');
    xlim([C1_freqBins(2) C1_freqBins(end)]);
    set(gca,'XTick',[C1_freqBins(2):2: C1_freqBins(end)]);
    %
    % ylim([1 size(C1Data,2)]);
    set(gca,'YTick',[1:1: size(C1Data,2)]);
    set(gca,'Yticklabel',char(dat.chan));
    % set(gca,'YTick',[freq_bins(4):2: req_bins(end-1)]);
    view(2);
    colormap jet;
    colorbar;
    caxis([0 200]);
    xlabel('Frequency [Hz]');
    ylabel('Channels');
    title('Frequency band-power per channel in Class 1');
    
    %% class 2
    for trial=1:size(C2.x,2)
        plotData2=C2.x(:,trial,:);
        plotData2=squeeze(plotData2);
        [trialspectrum, C2_freqBins] = mem( plotData2, params );
        tmp=mean(trialspectrum, 3);
        C2Data(:,:,trial)=tmp;
    end
    C2_freqBins = C2_freqBins - opt.freqBinWidth/2;
    C2plotData=mean(C2Data,3);
    C2plotData=cat(2, C2plotData, zeros(size(C2plotData, 1), 1));
    C2plotData=cat(1, C2plotData, zeros(1, size(C2plotData, 2)));
    C2_freqBins(end+1) = C2_freqBins(end) + diff(C2_freqBins(end-1:end));
    
    % Plot
    figure(2);
    surf(C2_freqBins, [1:size(C2plotData, 2)], C2plotData(1:end,:)');
    axis tight;
    set(gcf,'Renderer','Zbuffer');
    xlim([C2_freqBins(2) C2_freqBins(end)]);
    set(gca,'XTick',[C2_freqBins(2):2: C2_freqBins(end)]);
    %
    % ylim([1 size(C2Data,2)]);
    set(gca,'YTick',[1:1: size(C2Data,2)]);
    set(gca,'Yticklabel',char(dat.chan));
    % set(gca,'YTick',[freq_bins(4):2: req_bins(end-1)]);
    view(2);
    colormap jet;
    colorbar;
    caxis([0 200]);
    xlabel('Frequency [Hz]');
    ylabel('Channels');
    title('Frequency band-power per channel in Class 2');
    
    %% r square value
    C1Data = double(C1Data); C2Data = double(C2Data);
    for ch=1:size(C1Data, 2)
        for samp=1:size(C1Data, 1)
            ressq(samp, ch)=rsqu(C1Data(samp, ch, :), C2Data(samp, ch, :));
        end
    end
    ressq=cat(2, ressq, zeros(size(ressq, 1), 1));
    ressq=cat(1, ressq, zeros(1, size(ressq, 2)));
    ressq_freqBins = C2_freqBins;
    
    % Plot the r square value between classes
    figure(3);
    clf;
    surf(ressq_freqBins, [1:size(ressq, 2)], ressq(1:end,:)');
    
    axis tight;
    set(gcf,'Renderer','Zbuffer');
    xlim([ressq_freqBins(2) ressq_freqBins(end)]);
    set(gca,'XTick',[ressq_freqBins(2):2: ressq_freqBins(end)]);

    % ylim([1 size(ressq,2)]);
    set(gca,'YTick',[1:1: size(ressq,2)]);
    set(gca,'Yticklabel',char(dat.chan));
    % set(gca,'YTick',[freq_bins(4):2: req_bins(end-1)]);
    % view(2);
    view(2);
    colormap jet;
    colorbar;
    % caxis([0 150]);
    xlabel('Frequency [Hz]');
    ylabel('Channels');
    title('R-square vlaues per channel between classes using ');
end
end

