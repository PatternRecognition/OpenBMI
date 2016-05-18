function [ dat ] = plotERSP(data , varargin )
%UNTITLED 이 함수의 요약 설명 위치
%   자세한 설명 위치
%% It needs more modify code(?). please kindly wait until updating
%% Load the data
dat = data;
opt = opt_CellToStruct(varargin{:});
sampleFrequency=dat.fs;

%% Setting the parameter
% modelOrder = 18+round(sampleFrequency/100);
modelOrder = 20;
opt.lowPassCutoff=(sampleFrequency/2);
opt.freqBinWidth= 2;
opt.highPassCutoff=-1;
opt.trend=1;
% Setting the parameter about spectral moving (using autoregression)
opt.spectralSize= 500;
opt.spectralStep = 100;
opt.spectralSize = round(opt.spectralSize/1000 * sampleFrequency);
opt.spectralStep = round(opt.spectralStep/1000 * sampleFrequency);
params = [modelOrder, opt.highPassCutoff+opt.freqBinWidth/2, opt.lowPassCutoff, opt.freqBinWidth,round(opt.freqBinWidth/.2), ...
    opt.trend, sampleFrequency];
params(8) = opt.spectralStep;
params(9) = opt.spectralSize/opt.spectralStep;

%% Frequency-channel analysis
if strcmp(opt.Xaxis, 'Frequency') == 1 && strcmp(opt.Yaxis, 'Channel') == 1
    C1_idx=find(dat.y_dec==1);
    C2_idx = find(dat.y_dec==2);
    C1=prep_selectTrials(dat,C1_idx);
    C2=prep_selectTrials(dat,C2_idx);
    %% class 1 and class 2
    for trial=1:size(C1.x,2)
        plotData1=C1.x(:,trial,:);
        plotData1=squeeze(plotData1);
        [trialspectrum, C1_freqBins] = mem( plotData1, params );
        tmp=mean(trialspectrum, 3);
        C1Data(:,:,trial)=tmp;
        
        plotData2=C2.x(:,trial,:);
        plotData2=squeeze(plotData2);
        [trialspectrum2, C2_freqBins] = mem( plotData2, params );
        tmp2=mean(trialspectrum2, 3);
        C2Data(:,:,trial)=tmp2;
    end
    % class 1
    C1_freqBins = C1_freqBins - opt.freqBinWidth/2;
    C1plotData=mean(C1Data,3);  
    C1plotData=cat(2, C1plotData, zeros(size(C1plotData, 1), 1));
    C1plotData=cat(1, C1plotData, zeros(1, size(C1plotData, 2)));
    C1_freqBins(end+1) = C1_freqBins(end) + diff(C1_freqBins(end-1:end));
    % class 2
    C2_freqBins = C2_freqBins - opt.freqBinWidth/2;
    C2plotData=mean(C2Data,3);
    C2plotData=cat(2, C2plotData, zeros(size(C2plotData, 1), 1));
    C2plotData=cat(1, C2plotData, zeros(1, size(C2plotData, 2)));
    C2_freqBins(end+1) = C2_freqBins(end) + diff(C2_freqBins(end-1:end));
    % Plot
    figure(1);
    surf(C1_freqBins, [1:size(C1plotData, 2)], C1plotData(1:end,:)');
    axis tight;
    set(gcf,'Renderer','Zbuffer');
    
    xlim([C1_freqBins(2) C1_freqBins(end)]);
    set(gca,'XTick',[C1_freqBins(2):2: C1_freqBins(end)]);
    set(gca,'YTick',[1:1: size(C1Data,2)]);
    set(gca,'Yticklabel',char(dat.chan));
    view(2);
    colormap jet;
    colorbar;
    %modification of scale
    caxis([0 300]);
    xlabel('Frequency [Hz]');
    ylabel('Channels');
    title('Frequency band-power per channel in Class 1');
    
    figure(2);
    surf(C2_freqBins, [1:size(C2plotData, 2)], C2plotData(1:end,:)');
    axis tight;
    set(gcf,'Renderer','Zbuffer');
    xlim([C2_freqBins(2) C2_freqBins(end)]);
    set(gca,'XTick',[C2_freqBins(2):2: C2_freqBins(end)]);
    set(gca,'YTick',[1:1: size(C2Data,2)]);
    set(gca,'Yticklabel',char(dat.chan));

    view(2);
    colormap jet;
    colorbar;
    %modification of scale
    caxis([0 300]);
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
    set(gca,'YTick',[1:1: size(C2Data,2)]);
    set(gca,'Yticklabel',char(dat.chan));
    
    view(2);
    colormap jet;
    colorbar;
    % caxis([0 150]);
    xlabel('Frequency [Hz]');
    ylabel('Channels');
    title('Dynamic changes per channels between classes using r^2 values');

%% Time-channel analysis  
elseif strcmp(opt.Xaxis, 'Time') == 1 && strcmp(opt.Yaxis, 'Channel') == 1
    [b,a]= butter(2, [opt.Band]/sampleFrequency*2);
    dat= proc_filt(dat, b, a);  
    
    C1_idx=find(dat.y_dec==1);
    C2_idx = find(dat.y_dec==2);
    C1=prep_selectTrials(dat,C1_idx);
    C2=prep_selectTrials(dat,C2_idx);
    %% class 1 and class 2
    C1permute  = permute(C1.x ,[1 3 2]);
    C1data = reshape(C1permute, [size(C1permute,1)*size(C1permute,2) size(C1permute,3)]);
    C2permute  = permute(C2.x ,[1 3 2]);
    C2data = reshape(C2permute, [size(C2permute,1)*size(C2permute,2) size(C2permute,3)]);
    for t = 1: size(C1data,1)
        avgC1data(t,:) = mean(C1data(t,:));
        avgC2data(t,:) = mean(C2data(t,:));
    end
    avgC1data2 = reshape(avgC1data , [size(C1permute,1) size(C1.chan,2)]);
    plotC1data = avgC1data2';
    plotC1data=cat(2, plotC1data, zeros(size(plotC1data, 1), 1));
    plotC1data=cat(1, plotC1data, zeros(1, size(plotC1data, 2)));
    xData = C1.ival;
    xData(end+1) = xData(end) + diff(xData(end-1:end));
    
    avgC2data2 = reshape(avgC2data , [size(C2permute,1) size(C2.chan,2)]);
    plotC2data = avgC2data2';
    plotC2data=cat(2, plotC2data, zeros(size(plotC2data, 1), 1));
    plotC2data=cat(1, plotC2data, zeros(1, size(plotC2data, 2)));
    xData2 = C2.ival;
    xData2(end+1) = xData2(end) + diff(xData2(end-1:end));
    %plot
    figure(1);
    surf(xData, [1:size(plotC1data,1)], plotC1data,'EdgeColor','none');
    axis tight;
    set(gcf,'Renderer','Zbuffer');
    set(gca,'YTick',[1:1: size(plotC1data,1)]);
    set(gca,'Yticklabel',char(dat.chan));
    view(2);
    colormap jet;
    colorbar;
    caxis([-5 5]);
    %plot
    figure(2);
    surf(xData2, [1:size(plotC2data,1)], plotC2data,'EdgeColor','none');
    axis tight;
    set(gcf,'Renderer','Zbuffer');
    set(gca,'YTick',[1:1: size(plotC2data,1)]);
    set(gca,'Yticklabel',char(dat.chan));
    view(2);
    colormap jet;
    colorbar;
    caxis([-5 5]);
    %% r value
    C1permute = double(C1permute); C2permute = double(C2permute);
    for ch=1:size(C1permute, 2)
        for samp=1:size(C1permute, 1)
            ressq2(samp, ch)=rsqu(C1permute(samp, ch, :), C2permute(samp, ch , :));
        end
    end
    ressq2=cat(2, ressq2, zeros(size(ressq2, 1), 1));
    ressq2=cat(1, ressq2, zeros(1, size(ressq2, 2)));
    ressq2 = ressq2';
    xData3 = C1.ival;
    xData3(end+1) = xData3(end) + diff(xData3(end-1:end));
    figure(3);
    surf(xData3, [1:size(ressq2,1)], ressq2,'EdgeColor','none');
    axis tight;
    set(gcf,'Renderer','Zbuffer');
    set(gca,'YTick',[1:1: size(ressq2,1)]);
    set(gca,'Yticklabel',char(dat.chan));
    view(2);
    colormap jet;
    colorbar;
%% Time-frequency analysis
elseif strcmp(opt.Xaxis, 'Time') == 1 && strcmp(opt.Yaxis, 'Frequency') == 1
    Chanidx = find(strcmp(dat.chan, opt.Channel)== 1 );
    C1_idx=find(dat.y_dec==1);
    C2_idx = find(dat.y_dec==2);
    C1=prep_selectTrials(dat,C1_idx);
    C2=prep_selectTrials(dat,C2_idx);
    C1.x = C1.x(:,:,Chanidx);
    C2.x = C2.x(:,:,Chanidx);
    %% class 1 and 2
    if size(C1.x,2)==size(C2.x,2)
        for trial=1:size(C1.x,2)
            plotData1=C1.x(:,trial);
            plotData2=C2.x(:,trial);
            %         plotData1=squeeze(plotData1);
            [trialspectrum1, C1_freqBins] = mem( plotData1, params );
            [trialspectrum2, C2_freqBins] = mem( plotData2, params );
            C1data(:,trial,:)=trialspectrum1;
            C2data(:,trial,:)=trialspectrum2;
        end
        % class1
        plotC1data=mean(C1data,2);
        plotC1data=squeeze(plotC1data);
        C1_freqBins = C1_freqBins - opt.freqBinWidth/2;
        plotC1data=cat(2, plotC1data, zeros(size(plotC1data, 1), 1));
        plotC1data=cat(1, plotC1data, zeros(1, size(plotC1data, 2)));
        C1_freqBins(end+1) = C1_freqBins(end) + diff(C1_freqBins(end-1:end));
        %class2
        plotC2data=mean(C2data,2);
        plotC2data=squeeze(plotC2data);
        C2_freqBins = C2_freqBins - opt.freqBinWidth/2;
        plotC2data=cat(2, plotC2data, zeros(size(plotC2data, 1), 1));
        plotC2data=cat(1, plotC2data, zeros(1, size(plotC2data, 2)));
        C2_freqBins(end+1) = C2_freqBins(end) + diff(C2_freqBins(end-1:end));
        
        %Set the x axis using time information
        a = dat.ival(1)/100;
        b = size(plotC1data,2)-abs(dat.ival(1)/100)-1;
        a = a*100;
        b = b*100;
        
        % Plot
        figure(1);
        %     surf([-19:77], [-1 C1_freqBins(6:end)']', plotC1data(5:end,:));
        surf([a:100:b], C1_freqBins, plotC1data(1:end,:));
        caxis([0 1000]);
        set(gca,'xTick',[a:500:b]);
        axis tight;
        set(gcf,'Renderer','Zbuffer');
        view(2);
        colormap jet;
        colorbar;
        xlabel('Time [ms]');
        ylabel('Frequency [Hz]');
        title(['Time-frequency ERSP in channel ', char(opt.Channel),  ' and class 1']);
        
        figure(2);
        surf([a:100:b], C2_freqBins, plotC2data(1:end,:));
        caxis([0 1000]);
        set(gca,'xTick',[a:500:b]);
        axis tight;
        set(gcf,'Renderer','Zbuffer');
        view(2);
        colormap jet;
        colorbar;
        xlabel('Time [ms]');
        ylabel('Frequency [Hz]');
        title(['Time-frequency ERSP in channel ', char(opt.Channel),  ' and class 2']);
    end
end
