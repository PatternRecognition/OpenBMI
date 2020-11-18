%% CALIBRATION - multiClass grasping actions (realMove data only, EMG pattern classification approach) 
clear all; close all; clc;
tic

%% Load Data
dd = 'C:\Users\Doyeunlee\Desktop\Analysis\01_Raw data\';
filelist = 'session1_sub1_multigrasp_MI_EMG'; % this code is only for realMove data 

% for image dataset building
% order_count = 0; % from 0 ~  

%% Parameter setting for EEG
filt_band_eeg = [0.1 40]; % [0.3 3]Hz, [0.3 35]Hz -> reference: Decoding natural reach-and-grasp actions from human EEG 
% ival_rest = [-7000 -3000];
ival = [0 4000]; % for MOTOR IMAGERY % 250Hz sampling? Should besame for EEG and EMG 
% subChannel_eeg = [11 12 13 15 16 17 18 20 21 22 23 43 44 45 47 48 49 51 52 53]; %20 out of 60 channels
subChannel_eeg = [6 7 8 9 11 12 13 15 16 17 18 20 21 22 23 25 26 27 28 39 40 10 ...
    43 44 45 47 48 49 51 52 53 55 56 57]; %34 out of 60 channels
main_class = {'Cylindrical','Spherical','Lumbrical','Rest'};

% Parameter setting for EMG
filt_band_emg = [10 225]; % EMG [10 225]Hz  
subChannel_emg = [65:70]; % 4 EMG corresponding to hand and wrist movements only (grasp actions) 
% Total 71ch(60 EEG, 4 EOG, 7 EMG) <- 6 EMG channels [65:70] = 6 EMG 

%% load thres.mat; % threshold data of EEG and EMG signals, respectively.
ival_thres = [-6000 -3000]; 
bline_list = [0.75 1.0 1.25 1.5];

for bline_count = 1:size(bline_list, 2)
    bline_val = bline_list(bline_count); 
    [thres_EEG, thres_EMG] = thresCalib(dd, filelist, ival_thres, filt_band_eeg, filt_band_emg, ...
        subChannel_eeg, subChannel_emg, main_class); 

    %% converted data file to cnt - EMG, EEG all same  
    [cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist], 'fs', 500); % Load cnt, mrk, mnt variables to Matlab (grasp data) 
    % separate cnt into EEG's and EMG's 
    %for EEG 
    cnt_eeg = proc_commonAverageReference(cnt, subChannel_eeg); 
    cnt_eeg = proc_selectChannels(cnt_eeg, subChannel_eeg); % Channel Selection
    % for EMG 
%     cnt_emg = proc_commonAverageReference(cnt, subChannel_emg); 
    cnt_emg = proc_selectChannels(cnt_emg, subChannel_emg); 

    %% IIR Filter (4th butterworth filtter -> reference: Decoding natural reach-and-grasp actions) 
    %for EEG 
    cnt_eeg = proc_filtButter(cnt_eeg, 3, filt_band_eeg);  % Band-pass filtering
    
    % FIR FIlter - Option 1 
%     [y1, D1] = bandpass_computeGPU(cnt.x, filt_band, cnt.fs, 'ImpulseResponse', 'fir'); 
%     cnt.x = gather(y1);

    % FIR Filter - Option 2
%     Fs = cnt_eeg.fs;  % Sampling Frequency
%     N    = 16;     % Order
%     Fc1  = filt_band_eeg(1);    % First Cutoff Frequency
%     Fc2  = filt_band_eeg(2);    % Second C utoff Frequency
%     flag = 'scale';  % Sampling Flag
%     % Create the window vector for the design algorithm.
%     win = blackman(N+1);
%     % Calculate the coefficients using the FIR1 function.
%     b  = fir1(N, [Fc1 Fc2]/(Fs/2), 'bandpass', win, flag);
%     Hd = dfilt.dffir(b);
%     cnt_eeg.x = filter(Hd, cnt_eeg.x);
    
    %for EMG
%     cnt_emg = proc_filtButter(cnt_emg, 5, filt_band_emg);  % Band-pass filtering 

    %% MNT_ADAPTMONTAGE - Adapts an electrode montage to another electrode set
    mnt = mnt_adaptMontage(mnt, cnt); %MNT_ADAPTMONTAGE - Adapts an electrode montage to another electrode set

    %% Noise elimination 
    % cnt_eeg.x = medfilt1(cnt_eeg.x); % median values, to remove the spike noise on data 
    % cnt_emg.x = medfilt1(cnt_emg.x); 
    
    %% Normalization, rescale, smoothing filter 
%     cnt_eeg.x = normalize(cnt_eeg.x); 
%     cnt_emg.x = normalize(cnt_emg.x); 
%     cnt_eeg.x = hampel(cnt_eeg.x, 2000);
%     cnt_emg.x = hampel(cnt_emg.x, 2000); 

    %% cnt to epoch    
%     w_size = 1010; %(ms), window size 400, step size 150, overlap size = 250(ms) -> 25 segments  
%     step_size = 130;
%     overlap_size = w_size - step_size; % 250ms 
%     org_ival = ival; % [0 4000]= 4000ms
%     seg_length = abs(org_ival(2) - org_ival(1));
%     n_segment = ((seg_length - w_size)/(w_size - overlap_size)) + 1; % 25 segments
%     
%     h = 0;
%     for h = 1:ch

    k = 0; 
    for k = 1:size(main_class, 2) % repeat 3 classes 
            i = 0; 
            for i = 1:n_segment % repeat 25 segments
                s_point = ((i-1)*step_size) + org_ival(1);
                end_point = s_point + w_size; 
                sub_ival = [s_point end_point];

                % make epoch for EMG 
                epo_emg = cntToEpo(cnt_emg, mrk, sub_ival);
                epo_emg = proc_selectClasses(epo_emg, main_class{k}); 
                % make epoch for EEG
                epo_eeg = cntToEpo(cnt_eeg, mrk, sub_ival); 
                epo_eeg = proc_selectClasses(epo_eeg, main_class{k}); 

                for j = 1:50 % number of trials 
                    % create muscle patterns 
                    temp.muscle_fv = epo_emg.x(:, :, j); 
                    temp.ch1 = temp.muscle_fv(:, 1); temp.ch2 = temp.muscle_fv(:, 2);
                    temp.ch3 = temp.muscle_fv(:, 3); temp.ch4 = temp.muscle_fv(:, 4);
                    temp.ch5 = temp.muscle_fv(:, 5); temp.ch6 = temp.muscle_fv(:, 6);

                    temp.rms_fv(1) = rms(temp.ch1); temp.rms_fv(2) = rms(temp.ch2);
                    temp.rms_fv(3) = rms(temp.ch3); temp.rms_fv(4) = rms(temp.ch4);
                    temp.rms_fv(5) = rms(temp.ch5); temp.rms_fv(6) = rms(temp.ch6);
                             

                    % build EEG epoches per each class
                    max_size = size(epo_eeg.x(:, :, j), 1); 
                    temp.x{j}(1:max_size, :, i) = epo_eeg.x(:, :, j);

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

    end % for k 
end

X = abs(temp.muscle_fv);
X_1 = movmean(X, 30);
plot(X_1);
hold on;
xlim([0 4000]);


%     k = 0; 
%     for k = 1:3 
%         i = 0; temp.epo_x{k} = temp.x_all{k}{1}; 
%         temp.y{k} = emg_ptrn{k}(:, :, 1)'; % create label 
%         for i = 2:50
%             % epoched EEG data 
%             temp.epo_x{k} = cat(3, temp.epo_x{k}, temp.x_all{k}{i}); 
%             % labeling from EMG pattern 0, 1 
%             temp.y{k} = cat(2, temp.y{k}, emg_ptrn{k}(:, :, i)');  
% 
%         end
% 
%     end
% 
%     epo_all.x = cat(3, temp.epo_x{1}, temp.epo_x{2}, temp.epo_x{3}); 
%     epo_all.y = cat(2, temp.y{1}, temp.y{2}, temp.y{3}); 
% 
%     epo_all.className = {'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6'};
%     epo_all.clab = epo_eeg.clab; epo_all.fs = epo_eeg.fs; epo_all.title = epo_eeg.title; epo_all.file = epo_eeg.file; 
% 
%     % build epo
%     epo.x = epo_all.x; 
%     epo.clab = epo_all.clab; epo.fs = epo_all.fs; epo.title = epo_all.title; epo.file = epo_all.file; 
% 
%     %% repeat cross-validation for six times (classification results by channel) 
%     for n = 1:6
%         %% create epo by each class 
%         % One class vs. others (binary approach 6 times) 
%         epo.y = epo_all.y(n, :); epo.className = {'1', '0'}; % one of six channels 
% 
%         temp.epo_y = zeros(2, size(epo.y, 2)); 
%         temp.epo_y(1, :) = epo.y; 
%          for i = 1:size(epo.y, 2)
%             if temp.epo_y(1, i) == 1
%                 temp.epo_y(2, i) = 0;   
%             else
%                 temp.epo_y(2, i) = 1; 
% 
%             end
% 
%          end
%         epo.y = temp.epo_y; 
% 
%         %% eliminating NaN values
%         % epo.x = standardizeMissing(epo.x, -99);
%         % epo.x = fillmissing(epo.x, 'previous');    
% 
%         %% CSP feature extraction  
%         [csp_fv, csp_w{bline_count}{n}] = proc_csp3(epo); % apply CSP for binary classification 
% %         [csp_fv, csp_w{bline_count}{n}] = proc_csp_regularised(epo, 17, 0.7); % regulized CSP 
%         % % [csp_fv, csp_w] = proc_cspscp(epo_all, 2, 1); %CSP slow cortical potential variations 
% %         [csp_fv, csp_w{bline_count}{n}] = proc_csssp(epo, 17); % Common Sparse Spectrum Spatial Pattern   
% %         [csp_fv, csp_w] = proc_cspp_auto(epo); %auto csp patches, only for binary-class
%         csp_fv = proc_variance(csp_fv); 
%         csp_fv = proc_logarithm(csp_fv);
% 
%         %% all feature vectors 
%     %     all_csp_fv{n} = csp_fv; all_csp_w{n} = csp_w; 
% 
%         %% train classifier ver1
%         % N = length(epo.y); 
%         % idxTr = 1:N/2; 
%         % C = trainClassifier(csp_fv, 'LDA', idxTr);
% 
%         %% train classifier ver2 (Recommended)
%         csp_fv.classifier_param = {'LDA', 'prior', nan, 'store_prior', 1, 'store_means', 1, ...
%             'store_cov', 1, 'store_invcov', 1, 'scaling', 1};              
%         
%         proc = {'wr_multiClass','policy','one-vs-all','coding','hamming'}; % one-vs-all all-pairs
%         c_out{bline_count}{n}.C = trainClassifier(csp_fv, proc);
%         c_out{bline_count}{n}.out_eeg = applyClassifier(csp_fv, 'wr_multiClass', c_out{bline_count}{n}.C);
% 
%         %% apply classifier
%         [c_result{bline_count}{n}] = tnsre_applyClassifer(epo, c_out{bline_count}{n}, csp_w{bline_count}{n}); 
% 
%         %% check for classification accuracy each channel 
%         temp.true_label = epo.y(1, :); temp.test_label = c_result{bline_count}{n};  
%         temp.score = abs(temp.true_label - temp.test_label);
%         c_score{bline_count}{n} = 1 - (sum(temp.score)/size(epo.y, 2)); 
% 
%         %% Cross-validation for performance evaluation (Including CSP feature extraction process) 
%     %     proc = struct('memo', 'csp_w');
%     %     proc.train= ['[fv, csp_w, csp_eig]= proc_csp3(fv, 3); ' ...
%     %         'fv= proc_variance(fv); ' ...
%     %         'fv= proc_logarithm(fv);'];
%     %     proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
%     %         'fv= proc_variance(fv); ' ...
%     %         'fv= proc_logarithm(fv);'];
%     %     [C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(epo, 'LDA', 'proc', proc, 'kfold', 5); 
%     %     
%     %     % Result after cross validation = 1-error rate
%     %     Result{n} = 1 - C_eeg;
%     %     Result_Std{n} = loss_eeg_std;
% 
%     end
%     % % classification result from cross-validation test for each channel 
%     % Result
%     % Result_Std 
% 
%     % estimted EMG pattern from EEG data decoding 
%     esti_ptrn = vertcat(c_result{bline_count}{1}, c_result{bline_count}{2}, c_result{bline_count}{3}, c_result{bline_count}{4}, ...
%         c_result{bline_count}{5}, c_result{bline_count}{6}); 
% 
%     %% True label (true_ptrn) and test label (test_ptrn)
%     i = 0; s1 = 1; s2 = n_segment*50*1+1; s3 = n_segment*50*2+1;
%     true_l{bline_count} = epo_all.y; test_l{bline_count} = esti_ptrn; 
%     
% end
% %% save the trained classifier
% % % save C and csp_w (for version 1)
% % save('classifierVer1_out.mat', 'C', 'csp_w', 'filt_band_eeg', 'ival', 'subChannel_eeg', 'epo'); 
% % disp('Saving classifier is done!');
% 
% % save C, out_eeg (for version 2) 
% save classifier6emg_out.mat c_out csp_w; 
% % disp('Calibration work is done!'); 
% 
% %%
% i = 0; true_label = true_l{1}; test_label = test_l{1}; 
% for i = 2:bline_count 
%     true_label = vertcat(true_label, true_l{i}); 
%     test_label = vertcat(test_label, test_l{i}); 
%     
% end
% 
% for i = 1:50
%     nd1 = s1+(n_segment-1); nd2 = s2+(n_segment-1); nd3 = s3+(n_segment-1);   
%     % _ptrn{1} is for class 1 ... to 3
%     true_ptrn{1}(:, :, i) = true_label(:, s1:nd1);
%     true_ptrn{2}(:, :, i) = true_label(:, s2:nd2);
%     true_ptrn{3}(:, :, i) = true_label(:, s3:nd3);
%     
%     test_ptrn{1}(:, :, i) = test_label(:, s1:nd1); 
%     test_ptrn{2}(:, :, i) = test_label(:, s2:nd2);
%     test_ptrn{3}(:, :, i) = test_label(:, s3:nd3);
%     
%     s1 = nd1+1; s2 = nd2+1; s3 = nd3+1; 
%     
% end
% 
% %% save the dataset
% save truelabel6emg.mat true_label; 
% 
% %% test_ptrn to epo_ptrn dataset 
% i = 0; temp.epo_ptrn = cat(3, test_ptrn{1}, test_ptrn{2}, test_ptrn{3}); 
% for i = 1:150 
%     epo_ptrn.x(:, :, i) = temp.epo_ptrn(:, :, i)'; 
%     
% end
% epo_ptrn.y = zeros(3, 150); 
% epo_ptrn.y(1, 1:50) = 1; epo_ptrn.y(2, 51:100) = 1; epo_ptrn.y(3, 101:150) = 1; 
% epo_ptrn.className = main_class; 
% 
% %% ptrn matching test 
% class_num = 0; %1: cylindrical, 2: spherical, 3: lumbrical 
% for class_num = 1:3
%     j = 0; n = 0; i = 0; 
%     match_result = zeros(50, 3); match_val = zeros(1, 50);
%     for j = 1:50 % test_data trials 
%         for n = 1:3 
%             for i = 1:50 % trials 
%                 test_data = test_ptrn{class_num}(:, :, j); 
%                 ref_data = true_ptrn{n}(:, :, i); 
%                 % matching method (ssim, immse, psnr, ...) 
%                 error_val = immse(test_data, ref_data); 
% %                 match_val(i) = ssim(test_data, ref_data);
%                 match_val(i) = 1 - error_val; 
% 
%             end
%             match_result(j, n) = max(match_val);
% 
%         end
%     end
% 
%     [M, I] = max(match_result');
%     R_percent(class_num) = sum(I==class_num)/50; R(class_num) = mode(I);
% 
% end
% 
% R
% R_percent
% fianl_accuracy = sum(R_percent)/3
% 
% 
% %% save img_data (50 images for each class) 
% % create 36 by 36 size images 
% % n = 0; i = 0; s2 = '.jpg';
% % dir1 = '/Users/JEONG/Documents/MATLAB/robotArm_code/multiGraspDecoding/Online/temp_imgData/';
% % for n = 1:3
% %     dir2 = num2str(n-1); dir2 = strcat(dir2, '/'); % label folder starts from 0   
% %     for i = 1:50
% %         s1 = num2str(i+(50*order_count)); 
% %         filename = strcat(s1, s2); 
% %         dir = strcat(dir1, dir2);
% %         data = test_ptrn{n}(:, :, i); 
% %         imwrite(data, [dir filename]); 
% %     
% %     end
% % end
% % 
% % disp('Image saving is completed!'); 
% 
% toc



    
    