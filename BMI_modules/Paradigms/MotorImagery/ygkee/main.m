clc; clear all; close all;


%% eye movement or etc (smkim)


%% real movement(yjkee)
Makeparadigm_realmovement({'time_sti',5,'time_isi',3,'num_trial',4,'num_class',2,'time_jitter',0.1,'screenNumber',2})

%% Continous MI(yjkee)
Makeparadigm_realmovement({'time_sti',60,'time_isi',10,'num_trial',4,'num_class',2,'time_jitter',0.1,'screenNumber',2})

%% MI-no feedback (yjkee)
Makeparadigm_MI({'time_sti',5,'time_isi',3,'time_rest',3,'num_trial',50,'num_class',3,'time_jitter',0.1,'screenNumber',2})

%% MI(feedback)-mhlee

%% MI(feedback)-mhlee
