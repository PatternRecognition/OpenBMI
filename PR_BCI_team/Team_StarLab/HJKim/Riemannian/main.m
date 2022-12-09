%% Clear, close
clear; close; clc;

%% Data load
size.subject = 9;
size.trial = 288;
size.class = 4;
size.chan = 22;
size.sr = 250;

for i = 1:size.subject
    [train(i).s, train(i).h] = sload(char("C:\Users\KHJ-work\Documents\MATLAB\BCICIV_2a_gdf\A0" + int2str(i) + "T.gdf"));
    [eval(i).s, eval(i).h] = sload(char("C:\Users\KHJ-work\Documents\MATLAB\BCICIV_2a_gdf\A0" + int2str(i) + "E.gdf"));
    train(i).event_typ = find(train(i).h.EVENT.TYP >= 769 & train(i).h.EVENT.TYP <= 772 );
    eval(i).event_typ = find(eval(i).h.EVENT.TYP == 783);
    train(i).h.Classlabel = load(char("C:\Users\KHJ-work\Documents\MATLAB\BCICIV_2a_gdf\A0" + int2str(i) + "T.mat"));
    eval(i).h.Classlabel = load(char("C:\Users\KHJ-work\Documents\MATLAB\BCICIV_2a_gdf\A0" + int2str(i) + "E.mat"));
end

%% Segment
for i = 1:size.subject
    for j = 1:size.trial
        train(i).samp_seg(:, 1:size.chan, j) = train(i).s(train(i).h.TRIG(j) + 626:train(i).h.TRIG(j) + 1125, 1:size.chan);
        eval(i).samp_seg(:, 1:size.chan, j) = eval(i).s(eval(i).h.TRIG(j) + 626:eval(i).h.TRIG(j) + 1125, 1:size.chan);
        if j < size.trial
            train(i).ref_seg(:, 1:size.chan, j) = train(i).s(train(i).h.TRIG(j) + 1626:train(i).h.TRIG(j) + 2000, 1:size.chan);
            eval(i).ref_seg(:, 1:size.chan, j) = eval(i).s(eval(i).h.TRIG(j) + 1626:eval(i).h.TRIG(j) + 2000, 1:size.chan);
        end
    end
end

%% Remove NaN value trial
for i = 1:size.subject
    for j = 1:size.trial
        if find(isnan(train(i).samp_seg(:, :, j)))
            train(i).samp_seg(:, :, j) = zeros;
            train(i).samp_remain(j) = 0;
        else
            train(i).samp_remain(j) = 1;
        end
        
        if find(isnan(eval(i).samp_seg(:, :, j)))
            eval(i).samp_seg(:, :, j) = zeros;
            eval(i).samp_remain(j) = 0;
        else
            eval(i).samp_remain(j) = 1;
        end
        
        if j < size.trial
            if find(isnan(train(i).ref_seg(:, :, j)))
                train(i).ref_seg(:, :, j) = zeros;
                train(i).ref_remain(j) = 0;
            else
                train(i).ref_remain(j) = 1;
            end
            
            if find(isnan(eval(i).ref_seg(:, :, j)))
                eval(i).ref_seg(:, :, j) = zeros;
                eval(i).ref_remain(j) = 0;
            else
                eval(i).ref_remain(j) = 1;
            end
        end
    end
end

%% Bandpass filter
for i = 1:size.subject
    train(i).samp_segF = reshape(bandpass(reshape(train(i).samp_seg, [500 * size.trial size.chan]), [8 30], size.sr), [500, size.chan, size.trial]);
    eval(i).samp_segF = reshape(bandpass(reshape(eval(i).samp_seg, [500 * size.trial size.chan]), [8 30], size.sr), [500, size.chan, size.trial]);
    train(i).ref_segF = reshape(bandpass(reshape(train(i).ref_seg, [375 * (size.trial - 1) size.chan]), [8 30], size.sr), [375, size.chan, (size.trial - 1)]);
    eval(i).ref_segF = reshape(bandpass(reshape(eval(i).ref_seg, [375 * (size.trial - 1) size.chan]), [8 30], size.sr), [375, size.chan, (size.trial - 1)]);
end

%% Covariance matrix
for i = 1:size.subject
    for j = 1:size.trial
        train(i).samp_cov(:, :, j) = (1 / (length(find(train(i).samp_remain)) - 1)) * (train(i).samp_segF(:, :, j)' * train(i).samp_segF(:, :, j));
        eval(i).samp_cov(:, :, j) = (1 / (length(find(eval(i).samp_remain)) - 1)) * (eval(i).samp_segF(:, :, j)' * eval(i).samp_segF(:, :, j));
        if j < size.trial
            train(i).ref_cov(:, :, j) = (1 / (length(find(train(i).ref_remain)) - 1)) * (train(i).ref_segF(:, :, j)' * train(i).ref_segF(:, :, j));
            eval(i).ref_cov(:, :, j) = (1 / (length(find(eval(i).ref_remain)) - 1)) * (eval(i).ref_segF(:, :, j)' * eval(i).ref_segF(:, :, j));
        end
    end
end

%% Riemannian mean of whole sample and refference
for i = 1:size.subject
    train_samp_tmp = train(i).samp_cov;
    train_samp_tmp(:, :, train(i).samp_remain == 0) = [];
    eval_samp_tmp = eval(i).samp_cov;
    eval_samp_tmp(:, :, eval(i).samp_remain == 0) = [];
    train(i).samp_Rmean = mean(train_samp_tmp, 3);
    eval(i).samp_Rmean = mean(eval_samp_tmp, 3);
    
    train_ref_tmp = train(i).ref_cov;
    train_ref_tmp(:, :, train(i).ref_remain == 0) = [];
    eval_ref_tmp = eval(i).ref_cov;
    eval_ref_tmp(:, :, eval(i).ref_remain == 0) = []; 
    train(i).ref_Rmean = mean(train_ref_tmp, 3);
    eval(i).ref_Rmean = mean(eval_ref_tmp, 3);
    
    for j = 1:5
        Tsamp = tangentspace(train_samp_tmp, train(i).samp_Rmean);
        Tmean = mean(Tsamp, 3);
        train(i).samp_Rmean = untangentspace(Tmean, train(i).samp_Rmean);
        
        Tsamp = tangentspace(eval_samp_tmp, eval(i).samp_Rmean);
        Tmean = mean(Tsamp, 3);
        eval(i).samp_Rmean = untangentspace(Tmean, eval(i).samp_Rmean);
        
        Tsamp = tangentspace(train_samp_tmp, train(i).ref_Rmean);
        Tmean = mean(Tsamp, 3);
        train(i).ref_Rmean = untangentspace(Tmean, train(i).ref_Rmean);
        
        Tsamp = tangentspace(eval_samp_tmp, eval(i).ref_Rmean);
        Tmean = mean(Tsamp, 3);
        eval(i).ref_Rmean = untangentspace(Tmean, eval(i).ref_Rmean);
    end
end

%% Riemannian mean of each class
for i = 1:size.subject
    for j = 1:size.class
        train_samp_tmp = train(i).samp_cov;
        train_samp_tmp = train_samp_tmp(:, :, train(i).h.Classlabel.classlabel == j & train(i).samp_remain');
        eval_samp_tmp = eval(i).samp_cov;
        eval_samp_tmp = eval_samp_tmp(:, :, eval(i).h.Classlabel.classlabel == j & eval(i).samp_remain');
        
        train(i).class_Rmean(:, :, j) = mean(train_samp_tmp, 3);
        eval(i).class_Rmean(:, :, j) = mean(eval_samp_tmp, 3);

        for k = 1:5
            Tsamp = tangentspace(train_samp_tmp, train(i).class_Rmean(:, :, j));
            Tmean = mean(Tsamp, 3);
            train(i).class_Rmean(:, :, j) = untangentspace(Tmean, train(i).class_Rmean(:, :, j));
            
            Tsamp = tangentspace(eval_samp_tmp, eval(i).class_Rmean(:, :, j));
            Tmean = mean(Tsamp, 3);
            eval(i).class_Rmean(:, :, j) = untangentspace(Tmean, eval(i).class_Rmean(:, :, j));
        end
    end
end

%% Rmdm classification
for i = 1:size.subject
    tmp_res = Rmdm(eval(i).samp_cov(:, :, find(eval(i).samp_remain)), train(i).class_Rmean);
    acc1(i) = length(find(tmp_res == eval(i).h.Classlabel.classlabel(find(eval(i).samp_remain))')) / length(eval(i).samp_remain);
    tmp_res = Rmdm(train(i).samp_cov(:, :, find(train(i).samp_remain)), eval(i).class_Rmean);
    acc2(i) = length(find(tmp_res == train(i).h.Classlabel.classlabel(find(train(i).samp_remain))')) / length(train(i).samp_remain);
    def_acc(i) = (acc1(i) + acc2(i)) / 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Covariance matrix using affine transform
for i = 1:size.subject
    for j = 1:size.trial
        train(i).aff_cov(:, :, j) = (train(i).ref_Rmean ^ (-1 / 2)) * train(i).samp_cov(:, :, j) * ((train(i).ref_Rmean) ^ (-1 / 2));
        eval(i).aff_cov(:, :, j) = (eval(i).ref_Rmean ^ (-1 / 2)) * eval(i).samp_cov(:, :, j) * ((eval(i).ref_Rmean) ^ (-1 / 2));
    end
end

%% Riemannian mean of each class of affine transformed samples
for i = 1:size.subject
    for j = 1:size.class
        train_aff_tmp = train(i).aff_cov;
        train_aff_tmp = train_aff_tmp(:, :, train(i).h.Classlabel.classlabel == j & train(i).samp_remain');
        eval_aff_tmp = eval(i).aff_cov;
        eval_aff_tmp = eval_aff_tmp(:, :, eval(i).h.Classlabel.classlabel == j & eval(i).samp_remain');
        
        train(i).aff_class_Rmean(:, :, j) = mean(train_aff_tmp, 3);
        eval(i).aff_class_Rmean(:, :, j) = mean(eval_aff_tmp, 3);

        for k = 1:5
            Tsamp = tangentspace(train_aff_tmp, train(i).aff_class_Rmean(:, :, j));
            Tmean = mean(Tsamp, 3);
            train(i).aff_class_Rmean(:, :, j) = untangentspace(Tmean, train(i).aff_class_Rmean(:, :, j));
            
            Tsamp = tangentspace(eval_aff_tmp, eval(i).aff_class_Rmean(:, :, j));
            Tmean = mean(Tsamp, 3);
            eval(i).aff_class_Rmean(:, :, j) = untangentspace(Tmean, eval(i).aff_class_Rmean(:, :, j));
        end
    end
end

%% Rmdm classification of affine transformed samples
for i = 1:size.subject
    tmp_res = Rmdm(eval(i).aff_cov(:, :, find(eval(i).samp_remain)), train(i).aff_class_Rmean);
    acc1(i) = length(find(tmp_res == eval(i).h.Classlabel.classlabel(find(eval(i).samp_remain))')) / length(eval(i).samp_remain);
    tmp_res = Rmdm(train(i).aff_cov(:, :, find(train(i).samp_remain)), eval(i).aff_class_Rmean);
    acc2(i) = length(find(tmp_res == train(i).h.Classlabel.classlabel(find(train(i).samp_remain))')) / length(train(i).samp_remain);
    aff_acc(i) = (acc1(i) + acc2(i)) / 2;
end

%% MINE
for i = 1:size.subject
    for j = 1:size.subject
        for k = 1:size.class
            for l = find(train(j).h.Classlabel.classlabel' == k & train(j).samp_remain)
                tmp_samp(:, :, l) = (train(i).aff_class_Rmean(:, :, k) ^ (-1 / 2)) * train(j).aff_cov(:, :, l) * (train(i).aff_class_Rmean(:, :, k) ^ (-1 / 2));
            end
        end
        
        if i == 1 && j == 1
            train(i).transfer_sample = tmp_samp;
            continue
        end
        
        train(i).transfer_sample = cat(3, train(i).transfer_sample, tmp_samp);
    end
end

%%
for i = 1:size.subject
    if i == 1
        train_remain = train(i).samp_remain;
        train_class = train(i).h.Classlabel.classlabel;
       
    else
        train_remain = cat(2, train_remain, train(i).samp_remain);
        train_class = cat(1, train_class, train(i).h.Classlabel.classlabel);
    end
end

%%
for i = 1:size.subject
    for j = 1:size.class
        train_mine_tmp = train(i).transfer_sample;
        train_mine_tmp = train_mine_tmp(:, :, train_class == j & train_remain');
        
        train(i).mine_class_Rmean(:, :, j) = mean(train_mine_tmp, 3);
        
        for k = 1:5
            Tsamp = tangentspace(train_mine_tmp, train(i).mine_class_Rmean(:, :, j));
            Tmean = mean(Tsamp, 3);
            train(i).mine_class_Rmean(:, :, j) = untangentspace(Tmean, train(i).mine_class_Rmean(:, :, j));
        end
    end
end

%%
for i = 1:size.subject
    for j = 1:size.class
        for k = find(eval(j).h.Classlabel.classlabel' == j & eval(j).samp_remain)
            tmp_samp(:, :, k) = (train(i).aff_class_Rmean(:, :, j) ^ (-1 / 2)) * eval(i).aff_cov(:, :, k) * (train(i).aff_class_Rmean(:, :, j) ^ (-1 / 2));
        end
    end
    
    tmp_res = Rmdm(eval(i).aff_cov(:, :, find(eval(i).samp_remain)), train(i).mine_class_Rmean);
    acc1(i) = length(find(tmp_res == eval(i).h.Classlabel.classlabel(find(eval(i).samp_remain))')) / length(eval(i).samp_remain);
    %tmp_res = Rmdm(train(i).aff_cov(:, :, find(train(i).samp_remain)), eval(i).aff_class_Rmean);
    %acc2(i) = length(find(tmp_res == train(i).h.Classlabel.classlabel(find(train(i).samp_remain))')) / length(train(i).samp_remain);
    %aff_acc(i) = (acc1(i) + acc2(i)) / 2;
end