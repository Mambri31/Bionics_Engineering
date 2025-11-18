%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab grafica lo spettro di due oscillazioni cosinusoidali in rumore AWGN 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B=20;                        % Banda filtro passa-basso anti-aliasing in KHertz (KHz)
T= 100;                      % intervallo di osservazione misurato in millisecondi (msec)
A1=10;                       % Ampiezza del segnale 1
Theta1=2*pi/3;               % Fase iniziale in radianti segnale 1
f1=2;                        % frequenza analogica in KHertz (KHz) segnale 1
A2=3;                       % Ampiezza del segnale 2
Theta2=pi/3;                 % Fase iniziale in radianti segnale 2
f2=2+2/T;                    % frequenza analogica in KHertz (KHz) segnale 2
Tc=1/(2*B);                  % intervallo di campionamento misurato in millisecondi (msec)
N=round(2*B*T);              % Numero di campioni N=T/Tc=2BT deve essere >>1
Nzp=max(N,2^19);             % zero-padding per FFT
F1=f1*Tc;                    % frequenza digitale (adim) segnale 1 
F2=f2*Tc;                    % frequenza digitale (adim) segnale 1 
SNR1dBin=0;                  % Rapporto segnale_1-rumore in decibel [dB]
SNR1=10^(SNR1dBin/10);       % SNRin in scala lineare
N0=(A1^2)/(2*B*SNR1);        % Densità spettrale di Potenza monolatera del processo di rumore bianco  
varW=N0/2;                   % Potenza di rumore

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k=[0:N-1]';                                  % Vettore del tempo discreto
s1=A1*sqrt(1/(2*B))*cos(2*pi*F1*k+Theta1);   % generazione del vettore segnale 1 deterministico
s2=A2*sqrt(1/(2*B))*cos(2*pi*F2*k+Theta2);   % generazione del vettore segnale 2 deterministico
W=sqrt(varW)*randn(N,1);                     % generazione del vettore Nx1 di rumore
Y=s1+s2+W;                                       % generazione del vettore Nx1 dei dati osservati 
fx=2*B*(0:Nzp/2)'/Nzp;                       % Vettore delle frequenze analogiche (KHz)   
Yf=fft(Y,Nzp);
Yf=Yf(1:Nzp/2+1);
PeriodogramY=abs(Yf).^2./N;
Sf=fft(s1+s2,Nzp);
Sf=Sf(1:Nzp/2+1);
PeriodogramS=abs(Sf).^2./N;

% Grafico di una realizzazione del segnale osservato

Nc=100;            % Grafico dei primi Nc campioni
Nc=min(Nc,N);
Yfig=Y(1:Nc);               
tempo=k(1:Nc)*Tc;

figure(1)
grid on;
stem(tempo,Yfig,'b','LineWidth',2);
legend('X[n]');
title('First 100 samples of the observed signal [0,T)');xlabel('Time [msec]');ylabel('X[n]')

% Grafico del Periodogramma

figure (2);
plot(fx,10*log10(PeriodogramY),'LineWidth',2);grid on;
hold on
plot(fx,10*log10(PeriodogramS),'r--','LineWidth',1);
legend('Periodogram of observed data vector X','Periodogram of useful signal vector Ap');
title('Periodogram of observed and useful signal vectors');xlabel('Frequency (f) [KHz]');ylabel('Periodogram [dB]')
axis([0 20 -50 60])

% Zoom del Periodogramma intorno ad f0

figure (3);
plot(fx,10*log10(PeriodogramY),'LineWidth',2);grid on;
hold on
plot(fx,10*log10(PeriodogramS),'r--','LineWidth',1);
legend('Periodogram of observed data vector X','Periodogram of useful signal vector Ap');
title('Periodogram of observed and useful signal vectors');xlabel('Frequency (f) [KHz]');ylabel('Periodogram [dB]')
axis([1.9 2.1 -50 60])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 