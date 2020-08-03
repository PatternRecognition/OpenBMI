subdir = 'Cube_5';
conditions = {'Zoom', 'Tilt', 'Pan', 'Roll'};
marker_cond = [0 40 80 120];
marker_user_entry = 200;
marker_black      = 210;
marker_question   = 212;
marker_pressakey  = 214;
marker_video      = 220;

marker_start = 0;
marker_end   = 1;

% all parameters for all 4 conditions
% parameters = {
%   [1 1.0075 1.0150 1.0225 1.0300 1.0375 1.0450 1.0525 1.0600]
%   [0 0.115  0.229  0.344  0.458  0.573  0.688]
%   [0 0.458 -0.458  1.031 -1.031  1.490 -1.490  1.948 -1.948  2.979 -2.979  4.011 -4.011]
%   [0 0.458  1.031  1.490  1.948  2.979  4.011  5.042]
% };

parameters = {
  [1 1.0075 1.0150 1.0225 1.0300 1.0375 1.0450 1.0525 1.0600]
  [0 0.115  0.229  0.344  0.458  0.573  0.688]
  [0 0.458  1.031  1.490  1.948  2.979  4.011]
  [0 0.458  1.031  1.490  1.948  2.979  4.011  5.042]
};

kmax = 0;
for c=1:numel(conditions)
  kmax = kmax + numel(parameters{c});
end

file_names   = cell(kmax,1);
file_markers = zeros(kmax,1);

fmt_str = '%s\\%s\\cube_D0_V0_Z%.4f_Rx%.3f_Ry%.3f_Rz%.3f_G0_Vl1.00_Vr1.00_RD0.00_CAch0_CA0.000_EV0.00_W1R1.00_W3R1.00_00003';
k=0;
fid = fopen('file_list.csv','w');
for c=1:numel(conditions)
  p = {1, 0, 0, 0};
  for i=1:numel(parameters{c})
    k=k+1;
    p{c} = parameters{c}(i);
    file_names{k} = sprintf(fmt_str, conditions{c}, subdir, p{:});
    file_markers(k) = marker_cond(c) + 2*i;
    fprintf(fid, '%d, %s, %.5f, %s\n', file_markers(k), conditions{c}, parameters{c}(i), subdir);
  end
end
fclose(fid);

videos = textread('videos.csv','%s','delimiter','\n');
videos = reshape(videos,3,numel(videos)/3);
for i=1:size(videos,2)
  %convert mm:ss to seconds
  p = sscanf(videos{2,i}, '%d:%d');
  videos{2,i} = p(1)*60 + p(2);
end

clear fmt_str i c p k kmax fid;
