%% load synthetic data
load('arti_timeseries_100_2000_100.mat','XDat','Ktruev');
[n, p] = size(XDat);

%% compute Fourier transform and call BADGE
FDat = fft(XDat,[],1)/sqrt(n * 2 *pi);
FDat = FDat(1 : floor(n / 2) + 1, :);
[EKd_mat, EKod_mat, EJ_mat, Es_mat, run_time] = BADGE(FDat);

%% check performance
precision = sum(sum(Es_mat > 0.5)~=0 & sum(Ktruev~=0)~=0) / sum(sum(Es_mat > 0.5)~=0);
recall = sum(sum(Es_mat > 0.5)~=0 & sum(Ktruev~=0)~=0) / sum(sum(Ktruev~=0)~=0);
f1_score = 2*precision*recall/(precision+recall);
fprintf('precision = %d, recall = %d, f1-score = %d\n', precision, recall, f1_score); 