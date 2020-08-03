function bw2col(cols, hl)
%bw2col(cols, hl)
%
% reverse col2bw

for hi= 1:size(hl,1),
  set(hl(hi,1), 'color',cols(hl(hi,2),:), ...
                'lineWidth',hl(hi,3), 'lineStyle','-');
end
