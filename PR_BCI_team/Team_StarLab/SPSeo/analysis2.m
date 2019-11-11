function [ output_args ] = analysis2(varargin)

%% 
opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'segTime'),segTime=[-200 800];else segTime=opt.segTime;end
if ~isfield(opt,'baseTime'),baseTime=[-200 0];else baseTime=opt.baseTime;end
if ~isfield(opt,'selTime'),selTime=[0 800];else selTime=opt.selTime;end
if ~isfield(opt,'nFeature'),nFeature=10;else nFeature=opt.nFeature;end

if ~isfield(opt,'selectedFreq'),selectedFreq=[0.5 40];else selectedFreq=opt.selectedFreq;end



%% 

order_Dat = zeros(opt.total_run,6);
fileID2= fopen('save_order3.dat','w');
fwrite(fileID2, order_Dat,'double'); 
fclose(fileID2);
f_order2= memmapfile('save_order3.dat','Format',{'double' [15 72 6] 'x'} ,'Writable',true);
y_out2 = memmapfile('save_cly4.dat','Format',{'double' [15 72 37] 'x'} ,'Writable',true); 


notarestindx0or2 = zeros(1,1) 
fileID3 = fopen('save_notarestSP.dat','w');
fwrite(fileID3, notarestindx0or2,'double');
fclose(fileID3);
notarestSP = memmapfile('save_notarestSP.dat','Format', {'double' [1 1] 'x'},'Writable',true);




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

ival = [-200 400] * state.fs / 1000;


out_str = [];
output_str = [];
output_seg = [];
output_t = [];

clf_idx = 1;
seg_idx = 1;

cell_data = cell(num_char, 1);
clf_data = cell(num_char,1);



Y2 = [];
n_char = 1; 
%% run

while true 
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);

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
                if notarestSP.Data.x == 2; 
                    Y2 = []; 
                    n_char = n_char+1;
                    clf_idx = 1;
                end 
                if run_flag
                    run_flag = false;
                    output_t = [output_t t]; 
                    output_seg = cat(2, output_seg, seg_data); 
                    disp(size(seg_data,2)); 
                    
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
            
            org_data(:,11) = org_data(:,10);
            
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

            
            fv = func_featureExtraction(seg_data(:,clf_idx,:),{'feature','erpmean';'nMeans',nFeature}); 
            fv = reshape(squeeze(fv), [], 1); 

            
            ord_idx = f_order2.Data.x(n_char, clf_idx,:)
            [tm1 tm2] = find(ord_idx ~= 0);
            if ord_idx
                cell_data(ord_idx) = cellfun(@(x) horzcat(x, fv), cell_data(ord_idx), 'UniformOutput', false);
                clf_data(ord_idx) = cellfun(@(x) nanmean(x, 2), cell_data(ord_idx), 'UniformOutput', false);
                
                Y = func_predict([clf_data{:}], CF_PARAM_CLS);
                Y(end+1) = 1;
                y_out2.Data.x(n_char, clf_idx, :) = Y;

            else 
                ord_idx = ord_idx(tm2)
                cell_data(ord_idx) = cellfun(@(x) horzcat(x, fv), cell_data(ord_idx), 'UniformOutput', false);
                clf_data(ord_idx) = cellfun(@(x) nanmean(x, 2), cell_data(ord_idx), 'UniformOutput', false);
                
                Y = func_predict([clf_data{:}], CF_PARAM_CLS);
                Y(end+1) = 1;
                ord_idx(end+1) = 37;
                y_out2.Data.x(n_char, clf_idx, ord_idx) = Y(ord_idx);                

            end


            

            clf_idx = clf_idx + 1;
            
            fprintf('(%d) ·±,,, (%d) ½ÃÄö½º\n',n_run,n_seq) 
        end
    end
end

end

