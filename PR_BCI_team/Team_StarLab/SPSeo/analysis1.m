function [ output_args ] = analysis1(varargin)

%% Init
opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'segTime'),segTime=[-200 800];else segTime=opt.segTime;end
if ~isfield(opt,'baseTime'),baseTime=[-200 0];else baseTime=opt.baseTime;end
if ~isfield(opt,'selTime'),selTime=[0 800];else selTime=opt.selTime;end
if ~isfield(opt,'nFeature'),nFeature=10;else nFeature=opt.nFeature;end

if ~isfield(opt,'selectedFreq'),selectedFreq=[0.5 40];else selectedFreq=opt.selectedFreq;end

%% check online connection


sock = tcpip('localhost', 30000, 'NetworkRole', 'Client');
set(sock, 'OutputBufferSize', 1024); 

while(1)
    try
        fopen(sock);
        break;
    catch 
        fprintf('%s \n','Cant find Server');
    end
end
connectionSend = sock;


%% connection with brainvision
bbci_acquire_bv('close');
params = struct; 
state = bbci_acquire_bv('init', params); 

%% channels (index)
if isnumeric(opt.channel)
    channel_idx = false(1, length(state.clab)); 
    channel_idx(opt.channel) = true; 
    channel_idx = channel_idx(1:16); 
else
    channel_idx = ismember(state.clab, {'Cz', 'Oz'});  
end


seq = 6;
spr = 12;

%% classifier parameters
CF_PARAM = [];
CF_PARAM_CLS = opt.clf_param.classes; 

spell_char = ['A':'Z', '1':'9', '_']; 

num_char = length(spell_char);
order = importdata('C:\Users\cvpr\Desktop\Experiments\experiment_main\session3_files\random_order_v3_sp190730.mat'); 
run_flag = false;
ival = [-200 800] * state.fs / 1000;

out_str = [];
output_str = [];
output_seg = [];
output_t = [];

clf_idx = 1;
seg_idx = 1;

cell_data = cell(num_char, 1);
clf_data = cell(num_char,1);

n_charr = 1;
%% Visualization

%% run

while true 
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    % marker process
    for mrk_idx = 1:length(markertime)  
        mar = markerdescr(mrk_idx); 
        switch mar
            case 111
                disp('Paradigm Start');
            case {15, 16}  
                disp('Starting data segment');
                run_flag = true;
                org_data = [];
                seg_data = [];
                t = []; 
                seg_idx = 1;
                clf_idx = 1;
                
                tmp_data.x = [];
                [cell_data{:}] = deal([]); 
                [clf_data{:}] = deal(nan(nFeature*length(opt.channel),1));
                
            case 14 
                disp('Ending data segment');
                if run_flag
                    run_flag = false;
                    result = find(spell_char==out_str(end));
                    fprintf('Choose: %s\n', out_str(end));
                    fwrite(sock, result); 
                    output_str = [output_str out_str(end)]; 
                    output_t = [output_t t]; 
                    output_seg = cat(2, output_seg, seg_data); 
                    disp(size(seg_data,2));
                    
                    %%
                    if func_predict(clf_data{ind(1)}, CF_PARAM_CLS) < 0
                        disp('Passive');
                    else
                        disp('Active');
                    end
                    out_str = [];
                end

            case {1, 2, 3, 11}
                t = [t (markertime(mrk_idx)*state.fs/1000+length(org_data))]; 
            case {222, 20}
                output_args = {output_str, output_seg, output_t}; 
                return;
            case {77, 78, 18}
                continue;
        end
    end
    
    if run_flag && ~isempty(data) 
        org_data= [org_data; data(:,channel_idx)];
        
        while seg_idx <= length(t)
            if t(seg_idx)+ival(2) > size(org_data,1) 
                break;  
            end
            tmp_data.x = permute(org_data(t(seg_idx)+ival(1):t(seg_idx)+ival(2),:), [1 3 2]); 
            
            tmp_data.ival = linspace(ival(1), ival(2), ival(2) - ival(1) + 1); 
            tmp_data.fs = state.fs;
            tmp_data.chan = state.clab(channel_idx);
            tmp_data.t = t(seg_idx);
            
            tmp_data = prep_resample(tmp_data, 100);
            tmp_data = prep_baseline(tmp_data, {'Time', [-20 0]});
            seg_data(:,seg_idx,:) = tmp_data.x(find(tmp_data.ival == 0):end,:,:); 
            
            seg_idx = seg_idx + 1; 
        end
        
        while clf_idx < seg_idx 
            [n_run, n_seq] = ind2sub([spr, seq], clf_idx); 
                ord_idx = order{n_seq}(n_run, :);   
            fv = func_featureExtraction(seg_data(:,clf_idx,:),{'feature','erpmean';'nMeans',nFeature}); 
            fv = reshape(squeeze(fv), [], 1);

            cell_data(ord_idx) = cellfun(@(x) horzcat(x, fv), cell_data(ord_idx), 'UniformOutput', false); 
            clf_data(ord_idx) = cellfun(@(x) nanmean(x, 2), cell_data(ord_idx), 'UniformOutput', false);
       
            Y = func_predict([clf_data{:}], CF_PARAM_CLS); 
            [~, ind] = sort(Y,'ascend'); 

            out_str = [out_str, spell_char(ind(1))]; 
            
         
            
            clf_idx = clf_idx + 1;
            fprintf('(%d) 런,,, (%d) 시퀀스\n',n_run,n_seq)
            
            if n_run == 12 
 
                top4syminthisseq{n_charr,n_seq} = [ind(1) ind(2) ind(3) ind(4)]; 
                if n_seq == 6
                    n_charr = n_charr+1;
                end
                save('top4.mat','top4syminthisseq'); 
                save('n_seq.mat','n_seq');
            end                                
            fprintf('1등 %d,,, 2등 %d,,, 3등 %d,,, 4등 %d\n\n\n',ind(1),ind(2),ind(3),ind(4))
          


        end
    end
end
end
