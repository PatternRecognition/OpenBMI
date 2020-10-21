clc; clear all; close all;

dd = 'C:\Users\Doyeunlee\Desktop\Analysis\02_Converted_EMG\';
filelist={'session1_sub1_reaching_MI_EMG'};

main_class = {'Cylindrical','Spherical','Lumbrical','Rest'};
ival = [0 10000];
% filt_band_emg = [10 225];
subChannel_emg = [1:7];
ival_thres = [-1500 5000];
bline_list = [0.75 1.0 1.25 1.5];

for bline_count = 1:size(bline_list, 2)
    bline_val = bline_list(bline_count);
for i = 1:length(filelist)
    [cnt, mrk, mnt] = eegfile_loadMatlab([dd filelist{i}]);
    cnt = proc_commonAverageReference(cnt);
    thres_EMG = thresCalib(dd, filelist, ival_thres, [10 225], subChannel_emg, main_class);
    cnt = proc_filtButter(cnt, 5, [10 225]);
    epo = cntToEpo(cnt, mrk, ival);
    epo = proc_selectChannels(epo, {'EMG_1','EMG_2','EMG_3','EMG_4','EMG_5','EMG_6','EMG_ref'});
    mnt = mnt_adaptMontage(mnt,cnt);
    cnt.x = medfilt1(cnt.x);
    cnt.x = normalize(cnt.x);
    cnt.x = hampel(cnt.x, 2000);
    
    w_size = 1010; %(ms), window size 400, step size 150, overlap size = 250(ms) -> 25 segments
    step_size = 130;
    overlap_size = w_size - step_size; % 250ms
    org_ival = ival; % [0 4000]= 4000ms
    seg_length = abs(org_ival(2) - org_ival(1));
    n_segment = ((seg_length - w_size)/(w_size - overlap_size)) + 1; % 25 segments
    
    k = 0;
    for k = 1:size(main_class, 2)
        i = 0;
        for i = 1:n_segment
            s_point = ((i-1)*step_size) + org_ival(1);
            end_point = s_point + w_size;
            sub_ival = [s_point end_point];
            
            epo = cntToEpo(cnt, mrk, sub_ival);
            epo = proc_selectClasses(epo, main_class{k});
            
            for j = 1:50 % number of trials
                % create muscle patterns
                temp.muscle_fv = epo.x(:, :, j);
                temp.ch1 = temp.muscle_fv(:, 1); temp.ch2 = temp.muscle_fv(:, 2);
                temp.ch3 = temp.muscle_fv(:, 3); temp.ch4 = temp.muscle_fv(:, 4);
                temp.ch5 = temp.muscle_fv(:, 5); temp.ch6 = temp.muscle_fv(:, 6);
                
                temp.rms_fv(1) = rms(temp.ch1); temp.rms_fv(2) = rms(temp.ch2);
                temp.rms_fv(3) = rms(temp.ch3); temp.rms_fv(4) = rms(temp.ch4);
                temp.rms_fv(5) = rms(temp.ch5); temp.rms_fv(6) = rms(temp.ch6);
                
                ch_num = 0;
                for ch_num = 1:6 % number of channels
                    if temp.rms_fv(ch_num) > bline_val * thres_EMG{ch_num} % ???% of rest state rms(EMG value)
                        temp.emg_ptrn(i, ch_num, j) = 1;
                    else
                        temp.emg_ptrn(i, ch_num, j) = 0;
                        
                    end
                end
            end % for j
        end % for i
        emg_ptrn{k} = temp.emg_ptrn;
        temp.x_all{k} = temp.x;
    end
end
    k = 0;
    for k = 1:3
        i =0;
        temp.epo_x{k} = temp.x_all{k}{1};
        temp.y{k} = emg_ptrn{k}(:, :, 1)';
        for i = 2:50
            temp.y{k} = cat(2, temp.y{k}, emg_ptrn{k}(:,:, i)');
        end
    end
end    