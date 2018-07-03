function saved=loadEEG(dd, order, band, ival, channel_layout)

[cnt,mrk,mnt]=eegfile_loadMatlab(dd);
cnt = proc_filtButter(cnt,order,band);
cnt = proc_selectChannels(cnt,channel_layout);
% cnt to eo
epo=cntToEpo(cnt,mrk,ival);
%% variables can b changed by the data
classes=size(epo.className,2);
trial=size(epo.x,3)/2/(classes-1);

%% extract the rest class
for ii =1:classes
    if strcmp(epo.className{ii},'Rest')
        epoRest=proc_selectClasses(epo,{epo.className{ii}});
        epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
        epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
    else
        epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
        % random sampling
        epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
        epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
    end
end
if classes<7
    epo_check(size(epo_check,2)+1)=epoRest;
end

%% concatenate the classes
for ii=1:size(epo_check,2)
    if ii==1
        concatEpo=epo_check(ii);
    else
        concatEpo=proc_appendEpochs(concatEpo,epo_check(ii));
    end
end
saved=concatEpo;