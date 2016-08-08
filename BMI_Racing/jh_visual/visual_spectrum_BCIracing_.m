function [ dat ] = visual_spectrum_BCIracing_( data , varargin )
% visual_spectrum : It is the feature plots of data. This function can be
% useful to see an overview of data according to time, frequency, and
% channel. 
%
%
% % Synopsis:
%  [dat] = visual_spectrum(data , <OPT>)
%
% Example of synopsis about three types of domain:
%    visuspect = visual_spectrum(SMT , {'Xaxis' , 'Frequency'; 'Yaxis' , 'Channel'});
%    visuspect = visual_spectrum(SMT , {'Xaxis' , 'Time'; 'Yaxis' , 'Channel'; 'Band' ,[8 10]});
%    visuspect = visual_spectrum(SMT , {'Xaxis' , 'Time'; 'Yaxis' , 'Frequency'; 'Channel' ,{'C4'}});
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
%           G. Schalk, D.J. McFarland, T. Hinterberger, N. Birbaumer, and
%           J. R. Wolpaw,"BCI2000: A General-Purpose Brain-Computer
%           Interface (BCI) System, IEEE Transactions on Biomedical
%           Engineering, Vol. 51, No. 6, 2004, pp.1034-1043.
%
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
%

%% Load the data
dat = data;
opt = opt_cellToStruct(varargin{:});
sampleFrequency=dat.fs;

%% Setting the parameter
% We refer to visualization matlab code in BCI 2000 open source toolbox
%
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
if isnumeric(dat.class{1}) == 1 && isnumeric(dat.class{2}) == 1
    C1_idx=find(dat.y_dec==dat.class{1});
    C2_idx = find(dat.y_dec==dat.class{2});
else
    C1_idx=find(dat.y_dec==str2num(dat.class{1}));
    C2_idx = find(dat.y_dec==str2num(dat.class{2}));
end
% C1_idx=find(dat.y_dec==str2num(dat.class{1}));
% C2_idx = find(dat.y_dec==str2num(dat.class{2}));
C1=prep_selectTrials(dat,{'Index',C1_idx});
C2=prep_selectTrials(dat,{'Index',C2_idx});
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
    plotData = ressq;
    freqBins = C1_freqBins - opt.freqBinWidth/2;
    freqBins(end+1) = freqBins(end) + diff(freqBins(end-1:end));
  
        plotData=cat(2, plotData, zeros(size(plotData, 1), 1));
        plotData=cat(1, plotData, zeros(1, size(plotData, 2)));

        surf(freqBins, [1:size(plotData, 2)], plotData(1:end,:)'); hold on;
        axis tight; set(gcf,'Renderer','Zbuffer');
        xlim([freqBins(2) freqBins(end)]);
        set(gca,'XTick',[freqBins(2):4: freqBins(end)]);
        set(gca,'YTick',[1:1: size(C1Data,2)]);
        set(gca,'Yticklabel',char(dat.chan));
        str = dat.class{1,2};
        str2 = dat.class{2,2};  
        title([str,'  vs  ',str2]);
        view(2);colormap jet;colorbar;
        xlabel('Frequency [Hz]'); ylabel('Channels');
%         switch(i)
%             case 1
%                 caxis([0 300]);
%                 title('Frequency-channel power spectrum per channel in Class 1');
%             case 2
%                 caxis([0 300]);
%                 title('Frequency-channel power spectrum per channel in Class 2');
%             case 3
%                 title('Frequency-channel power spectrum per channels between classes using r^2 values');
%         end
end
%%     

end


