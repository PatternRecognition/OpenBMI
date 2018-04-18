% <<<3step>>> OVR approach를 위해 subject끼리 합치기
% -----------------------------------------------------------------------------------
% 1) loading files
% 2) merging
% 3) changing demensions
% -----------------------------------------------------------------------------------
 
fpath = ('D:\anedata_step\');
 
for i = 1:10 
    
    eval(['load d2_MM_' num2str(i) '.mat']);
    
end
 
MM_c = cat(3, d2_MM_1, d2_MM_2, d2_MM_3, d2_MM_5, d2_MM_6, d2_MM_7, d2_MM_8, d2_MM_9, d2_MM_10);
 
MM1 = MM_c(:, :, 1);
MM2 = MM_c(:, :, 2);
MM3 = MM_c(:, :, 3);
MM4 = MM_c(:, :, 4);
MM5 = MM_c(:, :, 4);
MM6 = MM_c(:, :, 5);
MM7 = MM_c(:, :, 6);
MM8 = MM_c(:, :, 7);
MM9 = MM_c(:, :, 8);
MM10 = MM_c(:, :, 9);
 
MM_all = cat(2, MM1, MM2, MM3, MM5, MM6, MM7, MM8, MM9, MM10);
 
save MM_3step_exceptMM4
