%  Brain connectivity

addpath('C:\Users\Doyeunlee\Desktop\Analysis\eeglab14_1_2b\');
eeglab;

srate = 250;
modIdx=2;
modality={'MI','realMove'};

range = {[4 8],[8 13],[13 40]};

filelist = {'sub01','sub02','sub03','sub04','sub05'};

for i = 1:size(range,2)
    for sub = 1: length(filelist)
        ival=[0 3000];
        [cntReach,mrkReach,mntReach]=eegfile_loadMatlab([dd modality{modIdx} '\' '250' '\' filelist{sub} '_reaching_' modality{modIdx}]);
        
        %Channel select
        temp_before=load_before.Data([],:,:);
        
        [wpli_before_temp] = WPLI2(temp_before,srate,range{i}(1),range{i}(2));
        
        wpli_before_temp = reshape(mean(wpli_before_temp,1),25,25);
        
        wpli_before(:,:,n)=wpli_before_temp+wpli_before_temp';
        
        
        
    end
end

