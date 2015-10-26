clc;
close all;
clear all;

%=========================================================================
%User-defined Parameters
%=========================================================================

tau = 30;          % MG parameter (delay)

N = 1100;          % no.  data points 

Ts = 6;            % sampling rate

dt = 0.01;         % inegration step 

%==========================================================================
% SNR in dB
%==========================================================================

snr = 10;

%==========================================================================

NN = N*Ts/dt;       % final number of samples

x = 0.9*ones(NN+1,1); % initial conditions

for n = tau/dt+1:NN,   
    
    xx = x(n);
    
    xxd = x(n - tau/dt);
 
    %=========================================================================
    % Runge-Kutta Method
    %=========================================================================
 
    xk1 = dt*(-0.1*xx + 0.2*xxd/(1+xxd^10));
    xk2 = dt*(-0.1*(xx+xk1/2) + 0.2*xxd/(1+xxd^10));
    xk3 = dt*(-0.1*(xx+xk2/2) + 0.2*xxd/(1+xxd^10));
    xk4 = dt*(-0.1*(xx+xk3) + 0.2*xxd/(1+xxd^10)); 

    x(n+1) = xx + xk1/6 + xk2/3 +xk3/3 +xk4/6;

end

%==========================================================================
% resample every Ts/dt
Xseq = x(Ts/dt+1:Ts/dt:NN);

Xseq = Xseq(100:end);

var_clean = var(Xseq);

sig = sqrt(var_clean/(10^(snr/10)));

%==========================================================================
% scale b/w [-1 1] to get the training (TN) signal normalized
%==========================================================================

ymax = 1; ymin = -1;

xmax = max(Xseq); xmin = min(Xseq);

TN_clean = (ymax-ymin)*(Xseq-xmin)/(xmax-xmin) + ymin;

save('C:\Haran\Research\NN\RMLP\NoisyMG\Web\TN_clean','TN_clean');

%==========================================================================

sigPwr = var(Xseq) +(mean(Xseq))^2;

sig = sqrt(sigPwr/(10^(snr/10)));

TN_noisy = TN_clean + sig*randn(length(TN_clean),1);

save('C:\Haran\Research\NN\RMLP\NoisyMG\Web\TN_noisy','TN_noisy');

tt_clean = TN_clean(801:900);

save('C:\Haran\Research\NN\RMLP\NoisyMG\Web\tt_clean','tt_clean');

tt_noisy = tt_clean + sig*randn(length(tt_clean),1);

save('C:\Haran\Research\NN\RMLP\NoisyMG\Web\tt_noisy','tt_noisy');

save('C:\Haran\Research\NN\RMLP\NoisyMG\Web\sig','sig');

%==========================================================================
figure;
plot(tt_clean,'r');
hold on;
plot(tt_noisy,'b:');
legend('Clean Test Signal','Noisy Test Signal',4);
