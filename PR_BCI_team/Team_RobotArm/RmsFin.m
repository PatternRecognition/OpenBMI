function rmsData=RmsFin(rmsWindow,rmsStride,foldedData)
%%
rmsData=foldedData;
for i =1:length(foldedData)
    %data is trial*channel*time
    data=permute(foldedData(i).x,[3 2 1]);
    [nTrials,nChannel,nTime]=size(data);
    for ii=1:rmsStride:nTime-rmsWindow
        window=zeros(nTrials,nChannel,rmsWindow);
        window=data(:,:,ii:ii+rmsWindow);
        %% rms
        windowSquare=window.*window;
        windowRms=sqrt(sum(windowSquare,3)/rmsWindow);
        if ii==1
            windowFin=windowRms;
            time=1;
        else
            time=cat(2,time,ii);
            windowFin=cat(3,windowFin,windowRms);
        end
    end
    windowFin=permute(windowFin,[3 2 1]);
    rmsData(i).x=windowFin;
    rmsData(i).t=time;
end
