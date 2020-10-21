clear all; close all; clc;

dd = 'C:\Users\Doyeunlee\Desktop\Analysis\01_Raw data\';
filelist = {'session1_sub1_reaching_MI_EMG'};
fold = 5;
ival = [0 4001];

selected_class = [1	2 3	4 5	6 7	8 9	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30	31 36	37	38	39	40	41	42	43	44	45	46	47	48	49	50	51	52	53	54	55	56	57	58	59	60	61	62	63	64];

for i = 1:length(filelist)
    [cnt, mrk, mnt] = eegfile_loadMatlab([dd filelist{i}], 'fs', 250);
    cnt = proc_filtButter(cnt, 2, [4 40]);
    cnt = proc_selectChannels(cnt, selected_class);
    epo = cntToEpo(cnt, mrk, ival);
    classes = size(epo.className, 2);
    trial = size(epo.x,3)/2/(classes-1);
    eachClassFold_no = trial/fold;
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
    epo_check = epo_check(1, 1:6);

    for ii=1:size(epo_check,2)
        if ii==1
            concatEpo=epo_check(ii);
        else
            concatEpo=proc_appendEpochs(concatEpo,epo_check(ii));
        end
    end
end

% visualization
ival_erd = [-500 4000];
ival_scalps = 0:1000:4000;
band = [4 40];
[b,a]=butter(5, band/cnt.fs*2);
cnt_flt=proc_channelwise(cnt, 'filtfilt',b,a);
ival = [-500 4000];
epo = concatEpo;
epo = cntToEpo(cnt, mrk, ival);
fv = proc_rectifyChannels(epo);
fv = proc_movingAverage(fv, 200, 'centered');
fv = proc_baseline(fv, [-500 0]);
erp = proc_average(fv);

% class º°·Î plotting
% plotClassTopographies(epo, mnt, ival);

% showERP(erp, mnt, cnt_flt);
scalpPlots(mnt, W);

% X = abs(a);
% X_1 = movmean(X, 30);
% plot(X_1);













    