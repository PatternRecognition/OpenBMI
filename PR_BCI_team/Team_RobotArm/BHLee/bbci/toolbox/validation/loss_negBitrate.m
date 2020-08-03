function loss= loss_negBitrate(label, out, seconds)
%loss= loss_negBitrate(label, out, <seconds>)
%
% IN  label - vector of true class labels (1...nClasses)
%     out   - array of classifier outputs
%     seconds - duration for acquiring all samples. if this argument is
%               given, the returned loss is given in bits per minute.
%
% OUT loss  - loss value (- bitrate)
%
% loss_negBitrate can handle outputs of classifiers that have a reject
% option (marked as NaN in the classifier output).
%
% SEE roc_curve

nClasses= size(label, 1);
valid= find(all(~isnan(out),1));
hit_prob= 1 - mean(loss_0_1(label(:,valid), out(:,valid)));
rate= bitrate(hit_prob, nClasses);

if nargin<3,
  factor= 1;
else
  factor= 60/length(valid)/seconds;
end

loss= -rate * length(valid) * factor;
