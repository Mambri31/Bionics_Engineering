%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab grafica lo spettro di un oscillazione cosinusoidale immersa in rumore AWGN 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A=10;                        % Ampiezza del segnale
Theta=2*pi/3;                % Fase iniziale in radianti
f0=2;                        % frequenza analogica in KHertz (KHz)
B=20;                        % Banda filtro passa-basso anti-aliasing in KHertz (KHz)
T= 100;                      % intervallo di osservazione misurato in millisecondi (msec)
Tc=1/(2*B);                  % intervallo di campionamento misurato in millisecondi (msec)
N=round(2*B*T);              % Numero di campioni N=T/Tc=2BT deve essere >>1
Nzp=max(N,2^19);             % zero-padding per FFT
F0=f0*Tc;                    % frequenza digitale (adim)                         
SNRdBin=0;                   % Rapporto segnale-rumore in decibel [dB]
SNR=10^(SNRdBin/10);         % SNRin in scala lineare
N0=(A^2)/(2*B*SNR);          % Densità spettrale di Potenza monolatera del processo di rumore bianco  
varW=N0/2;                   % Potenza di rumore

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k=[0:N-1]';                               % Vettore del tempo discreto
s=A*sqrt(1/(2*B))*cos(2*pi*F0*k+Theta);   % generazione del vettore segnale deterministico
W=sqrt(varW)*randn(N,1);                  % generazione del vettore Nx1 di rumore
Y=s+W;                                    % generazione del vettore Nx1 dei dati osservati 
fx=2*B*(0:Nzp/2)'/Nzp;                  % Vettore delle frequenze analogiche (KHz)   
Yf=fft(Y,Nzp);
Yf=Yf(1:Nzp/2+1);
PeriodogramY=abs(Yf).^2./N;
Sf=fft(s,Nzp);
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
 