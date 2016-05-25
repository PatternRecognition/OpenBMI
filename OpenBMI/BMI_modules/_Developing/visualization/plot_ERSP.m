function [ dat ] = plot_ERSP( data , varargin )
% ERSP : "Event-related spectral pergurbation" 
% Measuring the average dynamic changes in amplitude of the broad band
% EEG Frequency band
%
% % Synopsis:
%  [dat] = plot_ERSP(data , <OPT>)
%
% Example of synopsis about three types of domain
%    ersp = plot_ERSP(SMT , {'Xaxis' , 'Frequency'; 'Yaxis' , 'Channel'});
%    ersp = plot_ERSP(SMT , {'Xaxis' , 'Time'; 'Yaxis' , 'Channel'; 'Band' ,[8 10]});
%    ersp = plot_ERSP(SMT , {'Xaxis' , 'Time'; 'Yaxis' , 'Frequency'; 'Channel' ,{'C4'}});
%
% Arguments:
%   data: Data structrue (ex) Epoched data structure
%   <OPT> : 
%      .Xaxis - selecting the domain what you interested in x axis  
%                 (e.g. {'Xaxis' , 'Frequency'},{'Xaxis' , 'time'})
%      .Yaxis - selecting the domain what you interested in y axis  
%                 (e.g. {'Yaxis' , 'Channel'},{'Yaxis' , 'Frequency'})
%      .Band -  Selecting the interested frequency band in Time-Channel
%               domian
%                 (e.g. {'Band', [8 10]})
%      .Channel - Selecting the interested channel in Time-Frequency domain
%                 (e.g. {'Channel', {'C4'}})
%
% Return:
%    data:  Epoched data structure
%
% See also:
%    opt_cellToStruct
%
% Reference:
%
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
%

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
%%
C1_idx=find(dat.y_dec==1);
C2_idx = find(dat.y_dec==2);
C1=prep_selectTrials(dat,C1_idx);
C2=prep_selectTrials(dat,C2_idx);
for trial=1:size(C1.x,2)
    plotData1=C1.x(:,trial,:);
    plotData1=squeeze(plotData1);
    %trialspectrum = [frequency by channel by time by trial]
    [trialspectrum(:,:,:,trial), C1_freqBins] = mem( plotData1, params );
    plotData2=C2.x(:,trial,:);
    plotData2=squeeze(plotData2);
    [trialspectrum2(:,:,:,trial), C2_freqBins] = mem( plotData2, params );
end
%%
if strcmp(opt.Xaxis, 'Frequency') == 1 && strcmp(opt.Yaxis, 'Channel') == 1
    for tri = 1: size(C1_idx,2)
        tmp=mean(trialspectrum(:,:,:,tri),3);
        C1Data(:,:,tri)=tmp;
        tmp2=mean(trialspectrum2(:,:,:,tri),3);
        C2Data(:,:,tri)=tmp2;
    end
    C1Data = double(C1Data); C2Data = double(C2Data);
    for ch=1:size(C1Data, 2)
        for samp=1:size(C1Data, 1)
            ressq(samp, ch)=rsqu(C1Data(samp, ch, :), C2Data(samp, ch, :));
        end
    end
    plotData = {mean(C1Data,3),mean(C2Data,3), ressq};
    freqBins = C1_freqBins - opt.freqBinWidth/2;
    freqBins(end+1) = freqBins(end) + diff(freqBins(end-1:end));
    for i = 1:size(plotData,2)
        plotData{i}=cat(2, plotData{i}, zeros(size(plotData{i}, 1), 1));
        plotData{i}=cat(1, plotData{i}, zeros(1, size(plotData{i}, 2)));
        figure(i);
        surf(freqBins, [1:size(plotData{i}, 2)], plotData{i}(1:end,:)'); hold on;
        axis tight; set(gcf,'Renderer','Zbuffer');
        xlim([freqBins(2) freqBins(end)]);
        set(gca,'XTick',[freqBins(2):2: freqBins(end)]);
        set(gca,'YTick',[1:1: size(C1Data,2)]);
        set(gca,'Yticklabel',char(dat.chan));
        view(2);colormap jet;colorbar;
        xlabel('Frequency [Hz]'); ylabel('Channels');
        switch(i)
            case 1
                caxis([0 300]);
                title('Frequency band-power per channel in Class 1');
            case 2
                caxis([0 300]);
                title('Frequency band-power per channel in Class 2');
            case 3
                title('Dynamic changes per channels between classes using r^2 values');
        end
    end
%%     
elseif strcmp(opt.Xaxis, 'Time') == 1 && strcmp(opt.Yaxis, 'Channel') == 1
    fre_inx = find(C1_freqBins == opt.Band(2));
    for tri = 1: size(C1_idx,2)
        tmp = trialspectrum(fre_inx,:,:,tri);
        tmp = squeeze(tmp);
        C1Data(:,:,tri)=tmp;
        tmp2 = trialspectrum2(fre_inx,:,:,tri);
        tmp2 = squeeze(tmp2);
        C2Data(:,:,tri)=tmp2;
    end
    res1Data = double(permute(C1Data,[2 1 3]));
    res2Data = double(permute(C2Data,[2 1 3]));
    for ch=1:size(res1Data, 2)
        for samp=1:size(res1Data, 1)
            ressq(samp, ch)=rsqu(res1Data(samp, ch, :), res2Data(samp, ch, :));
        end
    end
    plotData = {mean(C1Data,3),mean(C2Data,3) , ressq'};
    a = dat.ival(1)/100;b = size(tmp,2)-abs(dat.ival(1)/100)-1;
    a = a*100;b = b*100;
    for i = 1:size(plotData,2)
        figure(i);
        surf([a:100:b], [1:size(plotData{i}, 1)], plotData{i}(1:end,:)); hold on;
        axis tight; set(gcf,'Renderer','Zbuffer');
        set(gca,'XTick',[a:500:b]);
        set(gca,'YTick',[1:1: size(C1Data,2)]);
        set(gca,'Yticklabel',char(dat.chan));
        view(2);colormap jet;colorbar;
        xlabel('Time'); ylabel('Channels');
        switch(i)
            case 1
                caxis([0 1000]);
                title('Frequency band-power per channel in Class 1');
            case 2
                caxis([0 1000]);
                title('Frequency band-power per channel in Class 2');
            case 3
                title('Dynamic changes per channels between classes using r^2 values');
        end
    end
%%
elseif strcmp(opt.Xaxis, 'Time') == 1 && strcmp(opt.Yaxis, 'Frequency') == 1
    Chanidx = find(strcmp(dat.chan, opt.Channel)== 1 );
    for tri = 1: size(C1_idx,2)
        tmp = trialspectrum(:,Chanidx,:,tri);
        tmp = squeeze(tmp);
        C1Data(:,:,tri)=tmp;
        tmp2 = trialspectrum2(:,Chanidx,:,tri);
        tmp2 = squeeze(tmp2);
        C2Data(:,:,tri)=tmp2;
    end
    plotData = {mean(C1Data,3),mean(C2Data,3)};
    a = dat.ival(1)/100;b = size(tmp,2)-abs(dat.ival(1)/100)-1;
    a = a*100;b = b*100;
    freqBins = C1_freqBins - opt.freqBinWidth/2;
    freqBins(end+1) = freqBins(end) + diff(freqBins(end-1:end));
    for i = 1:size(plotData,2)
        plotData{i}=cat(1, plotData{i}, zeros(1, size(plotData{i}, 2)));
        figure(i);
        surf([a:100:b]', freqBins', plotData{i}(1:end,:)); hold on;
        axis tight; set(gcf,'Renderer','Zbuffer');
        set(gca,'XTick',[a:500:b]);
        ylim([freqBins(2) freqBins(end)]);
        set(gca,'YTick',[freqBins(2):2: freqBins(end)]);
        view(2);colormap jet;colorbar;caxis([0 1000]);
        xlabel('Time'); ylabel('Frequency');
        switch(i)
            case 1
                title(['Time-frequency ERSP in channel ', char(opt.Channel),  ' and class 1']);
            case 2
                title(['Time-frequency ERSP in channel ', char(opt.Channel),  ' and class 2']);
        end
    end
end
end


