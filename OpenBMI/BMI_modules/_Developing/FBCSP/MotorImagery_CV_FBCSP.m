clc; clear all; close all;

dd= 'MotorImagery Converted Data\';
filelist= {'20160503_hblee_1'};
fold = 5;
ival = [500 3500];
% bnum = 7;   % filter bank number
% bandpassFilter = [4 9;9 14;14 19;19 24;24 29;29 34;34 39];
bnum = 9; % filter bank number
bandpassFilter = [4 8;8 12;12 16;16 20;20 24;24 28;28 32;32 36;36 40];
Result = []; L_Result= []; R_Result =[]; F_Result =[]; Rest_Result = [];

for i = 1:length(filelist)
    [cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist{i}]);
    cnt = proc_filterbank_hsan(cnt, 5, bandpassFilter);

        epo = cntToEpo(cnt, mrk, ival);
%     epo_prev = cntToEpo(cnt, mrk, ival);
% %     Reject the artifactual trials
%     [epo, iArte] = proc_rejectArtifactsTrials(epo_prev, ...
%         'clab', '*', 'ival', [], 'all_channels', 0,'verbose', 1);

    fResults = []; leftAcc = []; rightAcc = []; footAcc = []; restAcc = [];
    
    %% Find label indexes for each class
    leftLabel = []; rightLabel = []; footLabel = []; restLabel = [];
    for e = 1 : size(epo.x, 3)
        if epo.y(1, e)
            leftLabel = [leftLabel e];
        elseif epo.y(2, e)
            rightLabel = [rightLabel e];
        elseif epo.y(3, e)
            footLabel = [footLabel e];
        else
            restLabel = [restLabel e];
        end
    end
    lLength = floor(length(leftLabel) / fold); rLength = floor(length(rightLabel) / fold);
    fLength = floor(length(footLabel) / fold); restLength = floor(length(restLabel) / fold);
        
    for f = 1 : fold
        clear out_act out_mi;
        if f == fold
            lSt = (f - 1) * lLength + 1; rSt = (f - 1) * rLength + 1;
            fSt = (f - 1) * fLength + 1; restSt = (f - 1) * restLength + 1;
            epoIndex = [leftLabel(lSt : end) rightLabel(rSt : end) ...
                footLabel(fSt : end) restLabel(restSt : end)];
        else
            lSt = (f - 1) * lLength + 1; rSt = (f - 1) * rLength + 1;
            fSt = (f - 1) * fLength + 1; restSt = (f - 1) * restLength + 1;
            epoIndex = [leftLabel(lSt : lSt + lLength - 1) rightLabel(rSt : rSt + rLength - 1) ...
                footLabel(fSt : fSt + fLength - 1) restLabel(restSt : restSt + restLength - 1)];
        end
        epo_train = proc_selectEpochs(epo, 'not', epoIndex);
        epo_test = proc_selectEpochs(epo, epoIndex);
        
        %% Preparation for Check of Brain Activation
        epo_train_act = proc_combineClasses(epo_train, {'Left','Right','Foot'}, 'Rest');
        epo_train_act.y(1, find(epo_train_act.y(1, :) > 1)) = 1;
        
        [fv_train_act, out_act.old_csp_w, out_act.csp_w, actIdx] = proc_train_fbcsp(epo_train_act, bnum,0,0);
        out_act.C = trainClassifier(fv_train_act, {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});

        epo_test_act = proc_combineClasses(epo_test, {'Left','Right','Foot'}, 'Rest');
        epo_test_act.y(1, find(epo_test_act.y(1, :) > 1)) = 1;

        [fv_test_act] = proc_test_fbcsp(epo_test_act, out_act.old_csp_w, actIdx, bnum);
       
        out_act.results = applyClassifier(fv_test_act, 'RLDAshrink', out_act.C);
        out_act.results(find(out_act.results >= 0)) = 4;
        out_act.results(find(out_act.results < 0)) = 1;     

        %% Preparation for Classification of Motor Imagery
        epo_train_mi = proc_selectClasses(epo_train, 'not', 'Rest');
        
        [ fv_train_mi, out_mi.old_csp_w, out_mi.csp_w, miIdx1, miIdx2, miIdx3 ] = proc_train_fbcsp_mi(epo_train_mi, bnum,0,0);
        fv_train_mi.classifier_param = {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
                'store_cov', 1, 'store_invcov', 1, 'scaling', 1};
        proc = {'wr_multiClass_hsan','policy','one-vs-all','coding','hamming'}; % one-vs-all all-pairs
        
        out_mi.C = trainClassifier_hsan(fv_train_mi, proc, miIdx1, miIdx2, miIdx3);
        %% Classification of Motor Imagery for Brain Activated Labels
        for a = 1 : length(out_act.results)
            if out_act.results(a) == 1
                epo_test_mi = proc_selectEpochs(epo_test_act, a);
                [fv_test_mi] = proc_test_fbcsp_mi(epo_test_mi, out_mi.old_csp_w, miIdx1,miIdx2,miIdx3, bnum);
                fv_test_mi.x = fv_test_mi.x';
                
                tResult = applyClassifier_hsan(fv_test_mi, 'wr_multiClass_hsan', out_mi.C, miIdx1,miIdx2,miIdx3);
                out_act.results(a) = out2label(tResult);
            end
        end
        fResults(f) = sum(out2label(epo_test.y) == out_act.results) / length(epo_test.y) * 100;
        
        [mc, me] = calcConfusionMatrix(epo_test, out_act.results);
        leftAcc(f) = me(1,1);
        rightAcc(f) = me(2,2);
        footAcc(f) = me(3,3);
        restAcc(f) = me(4,4);
    end
    
    L_Result(i) = mean(leftAcc);
    R_Result(i) = mean(rightAcc);
    F_Result(i) = mean(footAcc);
    Rest_Result(i) = mean(restAcc);
    Result(i) = mean(fResults);
end


L_Result
R_Result
F_Result
Rest_Result
Result
mean(Result)