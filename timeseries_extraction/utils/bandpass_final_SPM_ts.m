function filter_ts = bandpass_final_SPM_ts(TR,fl,fh, ts)

% Sampling Frequency
Fs = 1/TR;

% Set bandpass filter parameters
% Center Frequency
Fc = 0.5*(fh + fl);
% Filter Order
No = floor(Fs * 2/Fc);
disp('Filtering ........................................................');
% ts is in the dimension of T*N
T = size(ts, 1);
if fh==0
	No=floor(T/4);
end
% FIR filter Coefficients
B = getFiltCoeffs(zeros(1,T),Fs,fl,fh,0,No);
A = 1;
% Filter the data
filter_ts = filtfilt(B,A,ts);
end
